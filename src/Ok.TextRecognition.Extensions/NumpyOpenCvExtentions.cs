using System;
using System.Collections.Generic;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Numpy;
using Numpy.Models;

namespace Ok.TextRecognition
{
    public static class NumpyOpenCvExtentions
    {
        public static Mat ToMatImage<T>(this NDarray npArrary)
        {
            var result = new Mat(npArrary.shape[0], npArrary.shape[1], GetDepthType<T>(), npArrary.shape.Dimensions.Length == 3 ? npArrary.shape[2] : 1);
            var arr = npArrary.GetData<T>();
            result.SetTo<T>(arr);
            return result;
        }

        public static NDarray ToImageNDarray<T>(this Mat mat)
        {
            return ToImageNDarray<T>(mat, mat.Cols, mat.Rows, mat.NumberOfChannels);
        }

        public static NDarray ToImageNDarray(this Mat mat, int width, int height, int channels)
        {
            return ToImageNDarray<float>(mat, width, height, channels);
        }

        public static NDarray ToImageNDarray<T>(this Mat mat, int width, int height, int channels)
        {
            var data = new T[height * width * mat.NumberOfChannels];
            mat.CopyTo<T>(data);
            return np.reshape(data, height, width, channels);
        }

        public static PointF[] ToPointsArray(this NDarray arr)
        {
            var points = new List<PointF>();
            for (var i = 0; i < arr.shape[0]; i++)
            {
                points.Add(new PointF((float)arr[i, 0], (float)arr[i, 1]));
            }

            return points.ToArray();
        }

        public static NDarray FromPointsArray(this PointF[] points)
        {
            var result = np.zeros(new Shape(points.Length, 2), np.float32);
            for (var i = 0; i < points.Length; i++)
            {
                result[i, 0] = (NDarray)points[i].X;
                result[i, 1] = (NDarray)points[i].Y;
            }

            return result;
        }        

        public static NDarray WhereFlags<T>(this NDarray input, NDarray flags, Func<bool, T, T> func)
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
    }
}
