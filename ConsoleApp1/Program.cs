using System;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Numpy;
using System.Runtime.InteropServices;
using Emgu.CV.Structure;
using Numpy.Models;

namespace ConsoleApp1
{
    class Program
    {
        public const int CC_STAT_AREA = 4;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            MLContext mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            var pipeline = mlContext.Transforms
                .LoadImages(
                    outputColumnName: "input.1", imageFolder: "assets",
                    inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill, outputColumnName: "input.1", imageWidth: 1184, imageHeight: 1280, inputColumnName: "input.1"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input.1"))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    modelFile: "craft.onnx",
                    outputColumnNames: new[] {
                               "273", "263"},
                    inputColumnNames: new[] {
                               "input.1"}));

            var mlNetModel = pipeline.Fit(dataView);

            var predictEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(mlNetModel);

            var result = predictEngine.Predict(new ImageNetData() { ImagePath = "1.jpg", Label= "1" });

            Mat scoreText = Mat.Ones(1, result.textmap.Length, DepthType.Cv32F, 1);
            scoreText.SetTo<float>(result.textmap);
            Mat scoreTextThresholded = new Mat();
            CvInvoke.Threshold(scoreText, scoreTextThresholded, 0.7, 1, ThresholdType.Binary);

            Mat scoreLink = Mat.Ones(1, result.linkmap.Length, DepthType.Cv32F, 1);
            scoreLink.SetTo<float>(result.linkmap);
            Mat scoreLinkThresholded = new Mat();
            CvInvoke.Threshold(scoreLink, scoreLinkThresholded, 0.7, 1, ThresholdType.Binary);

            var tempConcatArr = ToFloatVector(scoreTextThresholded).Concat(ToFloatVector(scoreLinkThresholded)).ToArray();

            var text_score_comb = np.clip(tempConcatArr, new float[] { 0 }, new float[] { 1 });

            var text_score_comb_bytes = text_score_comb.astype(np.uint8).GetData<byte>();

            var labels = new Mat();
            var statsMat = new Mat();
            var centroids = new Mat();
            var text_score_comb_data = Mat.Ones(1, text_score_comb_bytes.Length, DepthType.Cv8U, 1);
            text_score_comb_data.SetTo<byte>(text_score_comb_bytes);
            var nLabels = CvInvoke.ConnectedComponentsWithStats(text_score_comb_data, labels, statsMat, centroids, connectivity: LineType.FourConnected);

            var stats = ToIntMat(statsMat);
            for (var k = 0; k < nLabels; k++)
            {
                // size filtering
                var size = stats[k, CC_STAT_AREA];
                if (size < 10)
                {
                    continue;
                }

                // thresholding
                //if np.max(textmap[labels == k]) < text_threshold: continue

                //var segmap = np.zeros(new Shape(result.textmap), dtype: np.uint8);
            }
        }

        private static float[] ToFloatVector(Mat mat)
        {
            var result = new float[mat.Size.Width];

            var data = mat.GetData();
            int i = 0;
            foreach (var element in data)
            {
                result[i] = (float)element;
                i++;
            }

            return result;
        }

        private static int[,] ToIntMat(Mat mat)
        {
            var result = new int[mat.Rows, mat.Cols];

            var data = mat.GetData();
            int i = 0, j = 0;
            foreach (var element in data)
            {
                result[i, j] = (int)element;
                j++;
                if (j >= mat.Cols)
                {
                    i++;
                    j = 0;
                }
            }

            return result;
        }
    }
}
