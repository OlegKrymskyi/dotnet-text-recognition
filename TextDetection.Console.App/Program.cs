using System;
using System.Linq;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.Structure;
using Numpy;
using System.Drawing;
using Emgu.CV.CvEnum;

namespace TextDetection.App
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Start detection");
            
            var detector = new CraftTextDetector();

            var watch = new Stopwatch();
            watch.Start();
            using var result = detector.DetectTexts("assets/1.jpg");
            watch.Stop();

            var render_img = result.ScoreText.copy();
            render_img = np.hstack(render_img, result.ScoreLink);
            using var ret_score_text = render_img.Cvt2HeatmapImg();
            render_img.Dispose();

            using var rgbImage = new Mat("assets/1.jpg", loadType: ImreadModes.Color);
            CvInvoke.Imwrite("result_1_masked.jpg", ret_score_text);
            foreach (var idx in result.Boxes.Keys)
            {
                CvInvoke.Polylines(rgbImage, result.Boxes[idx].Select(x => new Point((int)(x.X), (int)(x.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 5);
            }

            CvInvoke.Imwrite("result_1.jpg", rgbImage);
        }
    }
}
