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
using SkiaSharp;
using System.Drawing;

namespace ConsoleApp1
{
    class Program
    {
        public const int CC_STAT_AREA = 4;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //using var img = (Bitmap)Bitmap.FromFile("assets/1.jpg");

            //var imgData = img.ToNDArray(flat: false, copy: false);

            MLContext mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            var pipeline = mlContext.Transforms
                .LoadImages(
                    outputColumnName: "image", imageFolder: "assets",
                    inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill, 
                    outputColumnName: "image", imageWidth: 1280, imageHeight: 1184, inputColumnName: "image"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "image"))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    modelFile: "craft-1280.onnx",
                    outputColumnNames: new[] {
                               "textmap", "linkmap"},
                    inputColumnNames: new[] {
                               "image"}));

            var mlNetModel = pipeline.Fit(dataView);

            var predictEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(mlNetModel);

            var result = predictEngine.Predict(new ImageNetData() { ImagePath = "1.jpg", Label= "1" });

            var textmapShape = np.array(result.textmap).shape;
            var img_h = textmapShape[0];
            var img_w = textmapShape[1];

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

                var textMapArr = new List<float>();
                var labelsTemp = new List<bool>();
                int i = 0;
                foreach (int label in labels.GetData())
                {
                    labelsTemp.Add(label == k);

                    if (label == k)
                    {
                        textMapArr.Add(result.textmap[i]);
                    }

                    i++;
                }


                // thresholding
                var textmapArr = new NDarray<float>(result.textmap);
                //textmapArr[labelsTemp];
                //if np.max(textmap[labels == k]) < text_threshold: continue

                if ((float)np.max(new NDarray<float>(textMapArr.ToArray())) < 0.7)
                {
                    continue;
                }

                //var segmap = np.zeros((new NDarray<float>(result.textmap)).shape, dtype: np.uint8);
                //i = 0;
                //foreach (int label in labels.GetData())
                //{
                //    if (label == k)
                //    {
                //        segmap[i] = (NDarray)255;
                //    }

                //    i++;
                //}

                var x = stats[k, (int)ConnectedComponentsTypes.Left];
                var y = stats[k, (int)ConnectedComponentsTypes.Top];
                var w = stats[k, (int)ConnectedComponentsTypes.Width];
                var h = stats[k, (int)ConnectedComponentsTypes.Height];

                var niter = (int)(Math.Sqrt(size * Math.Min(w, h) / (w * h)) * 2);
                var sx = x - niter;
                var ex = x + w + niter + 1;
                var sy = y - niter;
                var ey = y + h + niter + 1;

                if (sx < 0) sx = 0;
                if (sy < 0) sy = 0;
                if (ex >= img_w) ex = img_w;
                if (ey >= img_h) ey = img_h;
            }
        }

        private static Tuple<NDarray, float> resize_aspect_ratio(NDarray img, float square_size, Inter interpolation, float mag_ratio = 1)
        {
            var height = img.shape[0];
            var width = img.shape[1];
            var channel = img.shape[2];

            // magnify image size
            var target_size = mag_ratio * Math.Max(height, width);

            // set original image size
            if (target_size > square_size)
            {
                target_size = square_size;
            }

            var ratio = target_size / Math.Max(height, width);

            var target_h = (int)(height * ratio);
            var target_w = (int)(width * ratio);

            var imgMat = new Mat();
            CvInvoke.Resize(ToMatImage(img), imgMat, new Size(target_w, target_h), interpolation: interpolation);

            //# make canvas and paste image
            var target_h32 = target_h;
            var target_w32 = target_w;

            if (target_h % 32 != 0) {
                target_h32 = target_h + (32 - target_h % 32);
            }

            if (target_w % 32 != 0)
            {
                target_w32 = target_w + (32 - target_w % 32);
            }

            var resized = ToImageNDarray(imgMat, target_w32, target_h32, channel);

            return new Tuple<NDarray, float>(resized, ratio);
        }

        private static NDarray normalizeMeanVariance(NDarray in_img, NDarray mean, NDarray variance)
        {
            // should be RGB order
            var img = in_img.copy().astype(np.float32);

            img -= np.array(new NDarray[] { mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0 }, dtype: np.float32);
            img /= np.array(new NDarray[] { variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0 }, dtype: np.float32);
            
            return img;
        }

        private static NDarray cvt2HeatmapImg(NDarray img)
        {
            img = (np.clip(img, (NDarray)0, (NDarray)1) * 255).astype(np.uint8);
            var matImg = new Mat();
            CvInvoke.ApplyColorMap(ToMatImage(img), matImg, ColorMapType.Jet);
            return ToImageNDarray(matImg, matImg.Cols, matImg.Rows, matImg.NumberOfChannels);
        }

        private static Mat ToMatImage(NDarray npArrary)
        {
            var result = Mat.Eye(npArrary.shape[0], npArrary.shape[1], DepthType.Cv32F, npArrary.shape[2]);
            result.SetTo<float>(npArrary.GetData<float>());
            return result;
        }

        private static NDarray ToImageNDarray(Mat mat, int width, int height, int channels)
        {
            var resized = np.zeros((height, width, mat.NumberOfChannels), dtype: np.float32);

            var data = new float[height * width * mat.NumberOfChannels];
            mat.CopyTo<float>(data);
            for (var r = 0; r < height; r++)
            {
                for (var c = 0; c < width; c++)
                {
                    for (var i = 0; i < channels; i++)
                    {
                        resized[r, c, i] = (NDarray)data[r + c + i];
                    }
                }
            }
            return resized;
        }

        private static float[] ToFloatVector(Mat mat)
        {
            return ToVectorArray<float>(mat);
        }

        private static T[] ToVectorArray<T>(Mat mat)
        {
            var result = new T[mat.Size.Width];

            var data = mat.GetData();
            int i = 0;
            foreach (var element in data)
            {
                result[i] = (T)element;
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
