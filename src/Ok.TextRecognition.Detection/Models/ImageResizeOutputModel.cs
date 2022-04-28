using System;
using Numpy;

namespace Ok.TextRecognition.Detection.Models
{
    public class ImageResizeOutputModel: IDisposable
    {
        private bool disposed = false;

        public NDarray Image { get; set; }

        public float Ratio { get; set; }

        ~ImageResizeOutputModel()
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

            if (this.Image != null)
            {
                this.Image.Dispose();
            }

            this.disposed = true;
        }
    }
}
