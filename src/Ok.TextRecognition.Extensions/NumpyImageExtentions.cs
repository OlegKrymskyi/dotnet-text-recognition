using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Numpy;

namespace Ok.TextRecognition.Detection
{
    public static class NumpyImageExtentions
    {
        public static NDarray LoadRgbImage(this string filename)
        {
            using var imageMat = new Mat(filename, loadType: ImreadModes.Color);

            using var rgbImage = new Mat();
            CvInvoke.CvtColor(imageMat, rgbImage, ColorConversion.Bgr2Rgb);

            return rgbImage.ToImageNDarray<byte>();
        }

        public static PointF[] AdjustResultCoordinates(this PointF[] polys, float ratio_w, float ratio_h, float ratio_net = 2)
        {
            if (polys.Length > 0)
            {
                for (int k = 0; k < polys.Length; k++)
                {
                    polys[k].X *= ratio_w * ratio_net;
                    polys[k].Y *= ratio_h * ratio_net;
                }
            }

            return polys;
        }

        public static NDarray NormalizeMeanVariance(this NDarray in_img, NDarray mean, NDarray variance)
        {
            // should be RGB order
            var img = in_img.copy().astype(np.float32);

            img -= np.array(new NDarray[] { mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0 }, dtype: np.float32);
            img /= np.array(new NDarray[] { variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0 }, dtype: np.float32);

            return img;
        }

        public static Mat Cvt2HeatmapImg(this NDarray img)
        {
            img = (np.clip(img, (NDarray)0, (NDarray)1) * 255).astype(np.uint8);
            var matImg = new Mat();
            CvInvoke.ApplyColorMap(img.ToMatImage<byte>(), matImg, ColorMapType.Jet);
            return matImg;
        }
    }
}
