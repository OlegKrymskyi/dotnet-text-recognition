using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace Ok.TextRecognition.Recognition
{
    public class ColorBoundaryFinder
    {
        public RectangleF FindBoundary(Bitmap bitmap, byte min, byte max, int outset)
        {
            var rect = FindBoundary(bitmap, min, max);
            if (rect == RectangleF.Empty)
            {
                return RectangleF.Empty;
            }

            rect.X = rect.X - outset;
            rect.Y = rect.Y - outset;
            rect.Width = rect.Width + 2 * outset;
            rect.Height = rect.Height + 2 * outset;

            if (rect.X < 0)
            {
                rect.X = 0;
            }

            if (rect.Y < 0)
            {
                rect.Y = 0;
            }

            if (rect.X + rect.Width > bitmap.Width)
            {
                rect.Width = bitmap.Width - rect.X;
            }

            if (rect.Y + rect.Height > bitmap.Height)
            {
                rect.Height = bitmap.Height - rect.Y;
            }

            return rect;
        }

        public RectangleF FindBoundary(Bitmap bitmap, byte min, byte max)
        {
            using var mat = bitmap.ToImage<Gray, byte>();
            var rectangles =  FindBoundaries(mat, min, max);

            if (rectangles.Length == 0)
            {
                return RectangleF.Empty;
            }

            var maxRect = rectangles[0];
            for (var i = 1; i < rectangles.Length; i++)
            {
                maxRect = RectangleF.Union(maxRect, rectangles[i]);
            }

            return maxRect;
        }

        public RectangleF[] FindBoundaries(Bitmap bitmap, byte min, byte max)
        {
            using var mat = bitmap.ToImage<Gray, byte>();
            return FindBoundaries(mat, min, max);
        }

        public RectangleF[] FindBoundaries(Image<Gray, byte> img, byte min, byte max)
        {
            using var threadholdMat = img.InRange(new Gray(min), new Gray(max));

            using var contours = new Emgu.CV.Util.VectorOfVectorOfPoint();
            using var hierarchy = new Mat();
            CvInvoke.FindContours(threadholdMat, contours, hierarchy, RetrType.External, ChainApproxMethod.ChainApproxNone);

            var contoursPoints = contours.ToArrayOfArray();
            var rects = new List<RectangleF>();
            for (var i = 0; i < contoursPoints.Length; i++)
            {
                if (contoursPoints[i].Length >= 4)
                {
                    var area = MaxAreaRectangle(contoursPoints[i].Select(p => new PointF(p.X, p.Y)).ToArray());

                    if (area.Width > 1 && area.Height > 1)
                    {
                        rects.Add(area);
                    }
                }
            }

            return rects.ToArray();
        }

        private RectangleF MaxAreaRectangle(PointF[] points) 
        {
            if (points.Length == 0)
            {
                return RectangleF.Empty;
            }

            var maxX = points[0].X;
            var maxY = points[0].Y;

            var minX = points[0].X;
            var minY = points[0].Y;

            for (var i = 1; i < points.Length; i++)
            {
                if (minX > points[i].X) 
                {
                    minX = points[i].X;
                }

                if (minY > points[i].Y)
                {
                    minY = points[i].Y;
                }

                if (maxX < points[i].X)
                {
                    maxX = points[i].X;
                }

                if (maxY < points[i].Y)
                {
                    maxY = points[i].Y;
                }
            }

            return new RectangleF(new PointF(minX, minY), new SizeF(maxX - minX, maxY - minY));
        }
    }
}
