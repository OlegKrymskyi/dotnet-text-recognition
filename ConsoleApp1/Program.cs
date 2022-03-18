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

        public const float text_threshold = 0.7f;

        public const float link_threshold = 0.4f;

        public const float low_text = 0.4f;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            using var imageMat = new Mat("assets/1.jpg", loadType: ImreadModes.Color);

            using var rgbImage = new Mat();
            CvInvoke.CvtColor(imageMat, rgbImage, ColorConversion.Bgr2Rgb);

            using var inputData = ToImageNDarray<byte>(rgbImage);

            var resizedDataTuple = resize_aspect_ratio(inputData, 1280, Inter.Linear);

            //using var resizedImage = resizedDataTuple.Item1;

            MLContext mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

            var pipeline = mlContext.Transforms
                .LoadImages(
                    outputColumnName: "image", imageFolder: "assets",
                    inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill,
                    outputColumnName: "image", imageWidth: 1280, imageHeight: 1184))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "image"))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    modelFile: "craft-var.onnx",
                    outputColumnNames: new[] {
                               "output"},
                    inputColumnNames: new[] {
                               "image"}));

            var mlNetModel = pipeline.Fit(dataView);

            var predictEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(mlNetModel);

            var result = predictEngine.Predict(new ImageNetData() { ImagePath = "2.jpg" });

            using var outputArray = np.reshape(result.output, 1, 592, 640, 2);
            using var textmap = outputArray["0,:,:,0"];
            using var linkmap = outputArray["0,:,:,1"];

            var img_h = textmap.shape[0];
            var img_w = textmap.shape[1];

            using Mat textmapMat = ToMatImage<float>(textmap);
            using Mat textScoreThresholded = new Mat();
            CvInvoke.Threshold(textmapMat, textScoreThresholded, text_threshold, 1, ThresholdType.Binary);

            using Mat linkmapMat = ToMatImage<float>(linkmap);
            using Mat linkScoreThresholded = new Mat();
            CvInvoke.Threshold(linkmapMat, linkScoreThresholded, link_threshold, 1, ThresholdType.Binary);

            using var scoreText = ToImageNDarray<float>(textScoreThresholded);
            using var scoreLink = ToImageNDarray<float>(linkScoreThresholded);

            using var text_score_comb = np.clip(scoreText + scoreLink, (NDarray)0, (NDarray)1);

            using var text_score_comb_bytes = text_score_comb.astype(np.uint8);

            using var labelsMat = new Mat();
            using var statsMat = new Mat();
            using var centroidsMat = new Mat();
            using var text_score_comb_data = ToMatImage<byte>(text_score_comb_bytes);
            var nLabels = CvInvoke.ConnectedComponentsWithStats(text_score_comb_data, labelsMat, statsMat, centroidsMat, connectivity: LineType.FourConnected);

            using var stats = ToImageNDarray<int>(statsMat);
            using var labels = ToImageNDarray<int>(labelsMat);
            using var centroids = ToImageNDarray<float>(centroidsMat);
            for (var k = 1; k < nLabels; k++)
            {
                // size filtering
                var size = (int)stats[k, (int)ConnectedComponentsTypes.Area];
                if (size < 10)
                {
                    continue;
                }

                //using var labelFlags = SelectFlags<float>(labels, (elem) => elem == k);
                using var labelFlags = labels.equals(k);
                //using var textMapArr = textmap[labelFlags];
                using var textMapArr = WhereFlags<float>(textmap, labelFlags, (flag, elem) => flag ? elem : 0.0f);

                // thresholding
                if ((float)np.max(textMapArr) < text_threshold)
                {
                    continue;
                }

                // make segmentation map
                using var segmapZero = np.zeros(textmap.shape, dtype: np.uint8);
                using var segmap1 = WhereFlags<byte>(segmapZero, labelFlags, (flag, elem) => (byte)(flag ? 255 : 0));
                using var segmap = WhereFlags<byte>(segmap1, np.logical_and(scoreLink.equals(1), scoreText.equals(0)), (flag, elem) => (byte)(flag ? 0 : elem));

                var x = (int)stats[k, (int)ConnectedComponentsTypes.Left];
                var y = (int)stats[k, (int)ConnectedComponentsTypes.Top];
                var w = (int)stats[k, (int)ConnectedComponentsTypes.Width];
                var h = (int)stats[k, (int)ConnectedComponentsTypes.Height];

                var niter = (int)(Math.Sqrt(size * Math.Min(w, h) / (w * h)) * 2);
                var sx = x - niter;
                var ex = x + w + niter + 1;
                var sy = y - niter;
                var ey = y + h + niter + 1;

                // boundary check
                if (sx < 0) sx = 0;
                if (sy < 0) sy = 0;
                if (ex >= img_w) ex = img_w;
                if (ey >= img_h) ey = img_h;

                using var kernelMat = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(1 + niter, 1 + niter), new Point(-1,-1));
                using var dilateMat = new Mat();
                using var segmapMat = ToMatImage<byte>(segmap[$"{sy}:{ey},{sx}:{ex}"]);
                CvInvoke.Dilate(segmapMat, dilateMat, kernelMat, new Point(-1, -1), -1, BorderType.Default, new MCvScalar());

                using var dilate = ToImageNDarray<byte>(dilateMat);
                for (var i = sy; i < ey; i++)
                {
                    for (var j = sx; j < ex; j++)
                    {
                        segmap[i, j] = dilate[i - sy, j - sx];
                    }
                }

                // make box
                using var tempArr = np.roll(np.array(np.where(segmap.not_equals(0))), new int[] { 1 }, new Axis(0));
                using var np_contours = tempArr.transpose(0).reshape(-1, 2);
                //var rectangle = CvInvoke.MinAreaRect(np_contours);
                //var box = CvInvoke.BoxPoints(rectangle)
            }
        }

        public static NDarray SelectFlags<T>(NDarray input, Func<T, bool> expression)
        {
            var result = new List<bool>();
            foreach (var elem in input.GetData<T>())
            {
                result.Add(expression(elem));
            }

            using var flat = new NDarray<bool>(result.ToArray());

            return np.reshape(flat, input.shape);
        }

        public static NDarray WhereFlags<T>(NDarray input, NDarray flags, Func<bool, T, T> func)
        {
            var result = new List<T>();
            var data = input.GetData<T>();
            var flagsValues = flags.GetData<bool>();

            for (var i = 0; i < data.Length; i++)
            {
                result.Add(func(flagsValues[i], data[i]));
            }

            using var flat = new NDarray<T>(result.ToArray());

            return np.reshape(flat, input.shape);
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

            using var imgMat = new Mat();
            CvInvoke.Resize(ToMatImage<byte>(img), imgMat, new Size(target_w, target_h), interpolation: interpolation);

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

            var resized = ToImageNDarray<byte>(imgMat, target_w32, target_h32, channel);

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
            CvInvoke.ApplyColorMap(ToMatImage<float>(img), matImg, ColorMapType.Jet);
            return ToImageNDarray(matImg, matImg.Cols, matImg.Rows, matImg.NumberOfChannels);
        }

        private static Mat ToMatImage<T>(NDarray npArrary)
        {
            var result = new Mat(npArrary.shape[0], npArrary.shape[1], GetDepthType<T>(), npArrary.shape.Dimensions.Length == 3 ? npArrary.shape[2] : 1);
            var arr = npArrary.GetData<T>();
            result.SetTo<T>(arr);
            return result;
        }

        private static DepthType GetDepthType<T>()
        {
            if (typeof(T) == typeof(byte))
            {
                return DepthType.Cv8U;
            }
            else if (typeof(T) == typeof(float))
            {
                return DepthType.Cv32F;
            }

            return DepthType.Cv32F;
        }

        private static NDarray ToImageNDarray(Mat mat)
        {
            return ToImageNDarray(mat, mat.Cols, mat.Rows, mat.NumberOfChannels);
        }

        private static NDarray ToImageNDarray<T>(Mat mat)
        {
            return ToImageNDarray<T>(mat, mat.Cols, mat.Rows, mat.NumberOfChannels);
        }

        private static NDarray ToImageNDarray(Mat mat, int width, int height, int channels)
        {
            return ToImageNDarray<float>(mat, width, height, channels);
        }

        private static NDarray ToImageNDarray<T>(Mat mat, int width, int height, int channels)
        {
            var data = new T[height * width * mat.NumberOfChannels];
            mat.CopyTo<T>(data);
            return np.reshape(data, height, width, channels);
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
