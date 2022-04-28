using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Util;
using Ok.TextRecognition.Detection.Models;

namespace Ok.TextRecognition.Detection
{
    public class EastTextDetector: TextDetector
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

        public override TextDetectionResultModel DetectTexts(Mat imageMat)
        {
            var boxes = new VectorOfVectorOfPoint();
            var confidences = new VectorOfFloat();
            this.detector.Detect(imageMat, boxes, confidences);

            var result = new Dictionary<int, PointF[]>();
            for (var i = 0; i < boxes.Size; i++)
            {
                var points = new List<PointF>();
                for (var j = 0; j < boxes[i].Size; j++)
                {
                    points.Add(new PointF(boxes[i][j].X, boxes[i][j].Y));
                }

                result.Add(i + 1, points.ToArray());
            }

            return new TextDetectionResultModel
            {
                Boxes = result
            };
        }

        protected override void Release()
        {
            base.Release();
            this.detector.Dispose();
        }
    }
}
