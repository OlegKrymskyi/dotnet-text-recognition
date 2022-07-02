using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;

namespace Ok.TextRecognition.Recognition
{
    public class CrnnTextRecognizer: IDisposable
    {
        private bool disposed = false;

        private readonly TextRecognitionModel model;

        private readonly Size inputSize;

        public CrnnTextRecognizer(string modelFile = "assets/CRNN_VGG_BiLSTM_CTC.onnx", string vocabularyFile = "assets/alphabet_36.txt")
        {
            this.inputSize = new Size(100, 32);
            this.model = new TextRecognitionModel(modelFile);

            this.model.Vocabulary = File.ReadAllText(vocabularyFile).Split(new string[] { "\n" }, StringSplitOptions.RemoveEmptyEntries);
            this.model.DecodeType = "CTC-greedy";

            this.model.SetInputScale(1.0 / 127.5);
            this.model.SetInputMean(new MCvScalar(127.5, 127.5, 127.5));
            this.model.SetInputSize(this.inputSize);
        }

        ~CrnnTextRecognizer()
        {
            this.Dispose(false);
        }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (this.disposed)
            {
                return;
            }

            this.model.Dispose();
            this.disposed = true;
        }

        public string Recognize(Bitmap image, PointF[] box)
        {
            using var grayScale = image.ToImage<Gray, byte>();

            return Recognize(grayScale, box);
        }

        public string Recognize(Image<Gray, byte> image, PointF[] box)
        {
            return Recognize(image.Mat, box);
        }

        public string Recognize(Mat image, PointF[] box)
        {
            var targetVertices = new PointF[]
            {
                new PointF(0, this.inputSize.Height - 1),
                new PointF(0, 0),
                new PointF(this.inputSize.Width - 1, 0),
                new PointF(this.inputSize.Width - 1, this.inputSize.Height - 1),
            };
            using var rotationMatrix = CvInvoke.GetPerspectiveTransform(box, targetVertices);

            var cropped = new Mat();
            CvInvoke.WarpPerspective(image, cropped, rotationMatrix, this.inputSize);

            var result = model.Recognize(cropped);

            return result.Replace("\r", string.Empty);
        }
    }
}
