using System;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Numpy;
using TextRecognition;

namespace TextDetection.App
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start detection");

            var size = new Size(1280, 1280);
            using var imageMat = new Mat("assets/1.jpg", loadType: ImreadModes.Color);
            var ratioX = imageMat.Width / (float)size.Width;
            var ratioY = imageMat.Height / (float)size.Height;

            using var resizedMat = new Mat();
            CvInvoke.Resize(imageMat, resizedMat, size);

            var detector = new EastTextDetector(width: size.Width, height: size.Height);
            var recognizer = new CrnnTextRecognizer();

            var watch = new Stopwatch();
            watch.Start();
            using var result = detector.DetectTexts(resizedMat);
            watch.Stop();

            Console.WriteLine($"Detection took: {watch.Elapsed}");

            if (result.ScoreText != null)
            {
                var render_img = result.ScoreText.copy();
                render_img = np.hstack(render_img, result.ScoreLink);
                using var ret_score_text = render_img.Cvt2HeatmapImg();
                render_img.Dispose();

                CvInvoke.Imwrite("result_1_masked.jpg", ret_score_text);
            }

            using var grayImage = new Mat();
            CvInvoke.CvtColor(imageMat, grayImage, ColorConversion.Bgr2Gray);
            foreach (var idx in result.Boxes.Keys)
            {
                var points = result.Boxes[idx].Select(x => new PointF(x.X* ratioX, x.Y * ratioY)).ToArray();
                CvInvoke.Polylines(imageMat, points.Select(pt => new Point((int)(pt.X), (int)(pt.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 5);
                var text = recognizer.Recognize(grayImage, points);

                if (!string.IsNullOrWhiteSpace(text))
                {
                    Console.WriteLine($"Recognized text: {text}");
                }
            }

            CvInvoke.Imwrite("result_1.jpg", imageMat);
        }
    }
}
