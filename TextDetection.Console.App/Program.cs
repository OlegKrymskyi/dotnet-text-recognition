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
            
            var detector = new EastTextDetector();
            var recognizer = new CrnnTextRecognizer();

            var watch = new Stopwatch();
            watch.Start();
            using var result = detector.DetectTexts("assets/1.jpg");
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

            using var rgbImage = new Mat("assets/1.jpg", loadType: ImreadModes.Color);
            using var grayImage = new Mat("assets/1.jpg", loadType: ImreadModes.Grayscale);
            foreach (var idx in result.Boxes.Keys)
            {
                CvInvoke.Polylines(rgbImage, result.Boxes[idx].Select(x => new Point((int)(x.X), (int)(x.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 5);
                var text = recognizer.Recognize(grayImage, result.Boxes[idx]);

                if (!string.IsNullOrWhiteSpace(text))
                {
                    Console.WriteLine($"Recognized text: {text}");
                }
            }

            CvInvoke.Imwrite("result_1.jpg", rgbImage);
        }
    }
}
