using System;
using System.Collections.Generic;
using System.Drawing;
using Numpy;

namespace Ok.TextRecognition.Detection.Models
{
    public class TextDetectionResultModel : IDisposable
    {
        private bool disposed = false;

        public Dictionary<int, PointF[]> Boxes { get; set; }

        public NDarray ScoreText { get; set; }

        public NDarray ScoreLink { get; set; }

        ~TextDetectionResultModel()
        {
            this.Dispose(false);
        }

        public void Dispose()
        {
            this.Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (disposed)
            {
                return;
            }

            if (this.ScoreText != null)
            {
                this.ScoreText.Dispose();
            }

            if (this.ScoreLink != null)
            {
                this.ScoreLink.Dispose();
            }

            this.disposed = true;
        }
    }
}
