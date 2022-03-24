using System.Drawing;

namespace TextDetection
{
    public class ColorExtractor
    {
        public Bitmap ExtractColor(Bitmap bitmap, Color lower, Color upper)
        {
            var top = -1;
            var bottom = -1;

            for (var i = 0; i < bitmap.Height / 2; i++)
            {
                for (var j = 0; j < bitmap.Width; j++)
                {
                    var topPixel = bitmap.GetPixel(j, i);
                    var bottomPixel = bitmap.GetPixel(j, bitmap.Height - i - 1);

                    if (top == -1)
                    {
                        if (InColorInRange(topPixel, lower, upper))
                        {
                            top = i;
                        }
                    }

                    if (bottom == -1)
                    {
                        if (InColorInRange(bottomPixel, lower, upper))
                        {
                            bottom = bitmap.Height - i - 1;
                        }
                    }

                    if (top != -1 && bottom != -1)
                    {
                        break;
                    }
                }
            }

            var left = -1;
            var right = -1;
            for (var i = 0; i < bitmap.Height; i++)
            {
                for (var j = 0; j < bitmap.Width / 2; j++)
                {
                    var leftPixel = bitmap.GetPixel(j, i);
                    var rightPixel = bitmap.GetPixel(bitmap.Width - j - 1, i);

                    if (left == -1)
                    {
                        if (InColorInRange(leftPixel, lower, upper))
                        {
                            left = j;
                        }
                    }

                    if (right == -1)
                    {
                        if (InColorInRange(rightPixel, lower, upper))
                        {
                            right = bitmap.Width - j - 1;
                        }
                    }

                    if (left != -1 && right != -1)
                    {
                        break;
                    }
                }
            }

            if (right - left <= 0 || bottom - top <= 0)
            {
                return null;
            }

            var result = new Bitmap(right - left, bottom - top, System.Drawing.Imaging.PixelFormat.Format32bppRgb);
            using (var g = Graphics.FromImage(result))
            {
                g.DrawImage(bitmap, new Rectangle(0, 0, right - left, bottom - top), new Rectangle(top, left, right - left, bottom - top), GraphicsUnit.Pixel);
            }

            return result;
        }

        public bool InColorInRange(Color color, Color lower, Color upper)
        {
            return ((lower.R <= color.R) && (color.R <= upper.R)) &&
                ((lower.G <= color.G) && (color.G <= upper.G)) &&
                ((lower.B <= color.B) && (color.B <= upper.B));
        }
    }
}
