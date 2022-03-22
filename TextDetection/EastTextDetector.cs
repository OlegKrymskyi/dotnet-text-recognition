using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Util;
using TextDetection.Models;

namespace TextDetection
{
    public class EastTextDetector
    {
        private readonly TextDetectionModel_EAST detector;

        public EastTextDetector(string modelPath = "assets/frozen_east_text_detection.pb", int width = 1280, int height = 1280, float confidenceThreshold = 0.5f, float nmsThreshold = 0.4f)
        {
            this.detector = new TextDetectionModel_EAST(modelPath);
            this.detector.ConfidenceThreshold = confidenceThreshold;
            this.detector.NMSThreshold = nmsThreshold;
            this.detector.SetInputScale(1.0);
            this.detector.SetInputSize(new Size(width, height));
            this.detector.SetInputMean(new Emgu.CV.Structure.MCvScalar(123.68, 116.78, 103.94));
        }

        public TextDetectionResultModel DetectTexts(string imageFile)
        {
            using var imageMat = new Mat(imageFile, loadType: ImreadModes.Color);

            var result = new VectorOfVectorOfPoint();
            var confidences = new VectorOfFloat();
            this.detector.Detect(imageMat, result, confidences);

            return new TextDetectionResultModel
            {
            };
        }
    }
}
