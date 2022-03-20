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
using Microsoft.ML.Data;

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

            using var imageMat = new Mat("assets/test.png", loadType: ImreadModes.Color);

            using var rgbImage = new Mat();
            CvInvoke.CvtColor(imageMat, rgbImage, ColorConversion.Bgr2Rgb);

            using var inputData = ToImageNDarray<byte>(rgbImage);
            //var col0 = inputData["0,:,0"].copy();
            //var col1 = inputData["0,:,1"].copy();

            //inputData["0,:,1"] = col0;
            //inputData["0,:,0"] = col1;

            var resizedDataTuple = resize_aspect_ratio(inputData, 1280, Inter.Linear);

            var ratio_w = 1.0f / resizedDataTuple.Item2;
            var ratio_h = 1.0f / resizedDataTuple.Item2;

            using var resizedImage = resizedDataTuple.Item1;

            // preprocessing
            using var resizedImageNormalized = normalizeMeanVariance(resizedImage,
                mean: new NDarray(new float[] { 0.485f, 0.456f, 0.406f }),
                variance: new NDarray(new float[] { 0.229f, 0.224f, 0.225f }));

            // [h, w, c] to [c, h, w]
            using var finaleImage = resizedImageNormalized.transpose(2, 0, 1);

            MLContext mlContext = new MLContext();

            var dataView = mlContext.Data.LoadFromEnumerable(new List<ImageRawNetData>());

            var pipeline = mlContext.Transforms.ApplyOnnxModel(
                    modelFile: "craft-var.onnx",
                    outputColumnNames: new[] {
                               "output"},
                    inputColumnNames: new[] {
                               "image"});

            var mlNetModel = pipeline.Fit(dataView);

            var predictEngine = mlContext.Model.CreatePredictionEngine<ImageRawNetData, ImageNetPrediction>(mlNetModel);
            var result = predictEngine.Predict(new ImageRawNetData() { image = finaleImage.astype(np.float32).GetData<float>() });

            using var outputArray = np.reshape(result.output, 1, 592, 640, 2);
            using var textmap = outputArray["0,:,:,0"];
            using var linkmap = outputArray["0,:,:,1"];

            var img_h = textmap.shape[0];
            var img_w = textmap.shape[1];

            using Mat textmapMat = ToMatImage<float>(textmap);
            using Mat textScoreThresholded = new Mat();
            CvInvoke.Threshold(textmapMat, textScoreThresholded, low_text, 1, ThresholdType.Binary);

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
            var boxes = new Dictionary<int, PointF[]>();
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

                using var kernelMat = CvInvoke.GetStructuringElement(ElementShape.Rectangle, new Size(1 + niter, 1 + niter), new Point(-1, -1));
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
                using var tempArr = np.roll(np.array(np.where(segmap.not_equals(0))), new int[] { 1 }, axis: 0);
                using var np_contours = tempArr.transpose(1, 0).reshape(-1, 2);
                var rectangle = CvInvoke.MinAreaRect(ToPointsArray(np_contours));
                var boxPoints = CvInvoke.BoxPoints(rectangle);
                var box = FromPointsArray(boxPoints);

                // align diamond-shape
                var boxW = (float)np.linalg.norm(box[0] - box[1]);
                var boxH = (float)np.linalg.norm(box[1] - box[2]);
                var box_ratio = Math.Max(boxW, boxH) / (Math.Min(boxW, boxH) + 1e-5);
                if (Math.Abs(1 - box_ratio) <= 0.1)
                {
                    var l = (float)np.min(np_contours[":,0"]);
                    var r = (float)np.max(np_contours[":,0"]);
                    var t = (float)np.min(np_contours[":,1"]);
                    var b = (float)np.max(np_contours[":,1"]);
                    box = np.array(new float[,] { { l, t }, { r, t }, { r, b }, { l, b } }, dtype: np.float32);
                }

                // make clock-wise order
                var startidx = (int)box.sum(axis: 1).argmin();
                box = np.roll(box, new int[] { 4 - startidx }, axis: 0);
                box = np.array(box);
                boxes.Add(k, AdjustResultCoordinates(ToPointsArray(box), ratio_w, ratio_h));
                box.Dispose();
            }

            // render results (optional)
            var render_img = scoreText.copy();
            render_img = np.hstack(render_img, scoreLink);
            using var ret_score_text = cvt2HeatmapImg(render_img);
            render_img.Dispose();

            CvInvoke.Imwrite("result_2_masked.jpg", ret_score_text);
            foreach (var idx in boxes.Keys)
            {
                CvInvoke.Polylines(rgbImage, boxes[idx].Select(x => new Point((int)(x.X), (int)(x.Y))).ToArray(), true, new MCvScalar(255, 0, 0), thickness: 5);
            }

            CvInvoke.Imwrite("result_2.jpg", rgbImage);
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

            var ratio = target_size / (float)Math.Max(height, width);

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

        private static Mat cvt2HeatmapImg(NDarray img)
        {
            img = (np.clip(img, (NDarray)0, (NDarray)1) * 255).astype(np.uint8);
            var matImg = new Mat();
            CvInvoke.ApplyColorMap(ToMatImage<byte>(img), matImg, ColorMapType.Jet);
            return matImg;
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

        private static PointF[] ToPointsArray(NDarray arr)
        {
            var points = new List<PointF>();
            for (var i = 0; i < arr.shape[0]; i++)
            {
                points.Add(new PointF((float)arr[i, 0], (float)arr[i, 1]));
            }

            return points.ToArray();
        }

        private static NDarray FromPointsArray(PointF[] points)
        {
            var result = np.zeros(new Shape(points.Length, 2), np.float32);
            for (var i = 0; i < points.Length; i++)
            {
                result[i, 0] = (NDarray)points[i].X;
                result[i, 1] = (NDarray)points[i].Y;
            }

            return result;
        }

        private static PointF[] AdjustResultCoordinates(PointF[] polys, float ratio_w, float ratio_h, float ratio_net = 2)
        {
            if (polys.Length > 0)
            {
                for (int k = 0; k<polys.Length; k++)
                {
                    polys[k].X *= ratio_w * ratio_net;
                    polys[k].Y *= ratio_h * ratio_net;
                }
            }

            return polys;
        }
    }
}
