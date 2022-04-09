using System.Drawing;
using System.Drawing.Imaging;
using Emgu.CV;
using Emgu.CV.Structure;

namespace TextDetection
{
    public static class OpenCvExtension
    {
        public unsafe static Mat FromBitmap(this Bitmap bitmap)
        {
            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);
            var imageCV = new Image<Rgb, byte>(bitmap.Width, bitmap.Height, bitmapData.Stride, bitmapData.Scan0);
            bitmap.UnlockBits(bitmapData);

            return imageCV.Mat;
        }
    }
}
