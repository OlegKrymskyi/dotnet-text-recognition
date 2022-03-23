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
            
            var detector = new EastTextDetector(width: 1280, height: 1280);
            var recognizer = new CrnnTextRecognizer();

            using var imageMat = new Mat("assets/data/screen.png", loadType: ImreadModes.Color);
            using var resizedMat = new Mat();
            CvInvoke.Resize(imageMat, resizedMat, new Size(1280, 1280));

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
            CvInvoke.CvtColor(resizedMat, grayImage, ColorConversion.Bgr2Gray);
            foreach (var idx in result.Boxes.Keys)
            {
                CvInvoke.Polylines(resizedMat, result.Boxes[idx].Select(x => new Point((int)(x.X), (int)(x.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 5);
                var text = recognizer.Recognize(grayImage, result.Boxes[idx]);

                if (!string.IsNullOrWhiteSpace(text))
                {
                    Console.WriteLine($"Recognized text: {text}");
                }
            }

            CvInvoke.Imwrite("result_1.jpg", resizedMat);
        }
    }
}
