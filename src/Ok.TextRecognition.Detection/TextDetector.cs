using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Ok.TextRecognition.Detection.Models;

namespace Ok.TextRecognition.Detection
{
    public abstract class TextDetector
    {
        protected bool disposed = false;

        public TextDetectionResultModel DetectTexts(string imageFile)
        {
            using var imageMat = new Mat(imageFile, loadType: ImreadModes.Color);

            return DetectTexts(imageMat);
        }

        public TextDetectionResultModel DetectTexts(Bitmap bitmap)
        {
            using var imageMat = bitmap.ToImage<Bgr, byte>();

            return DetectTexts(imageMat.Mat);
        }

        public abstract TextDetectionResultModel DetectTexts(Mat imageMat);

        ~TextDetector()
        {
            this.Dispose(false);
        }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        public void Dispose(bool disposing)
        {
            if (this.disposed)
            {
                return;
            }

            this.Release();

            this.disposed = true;
        }

        protected virtual void Release()
        { 
        }
    }
}
