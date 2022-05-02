using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using FluentAssertions;
using Xunit;

namespace Ok.TextRecognition.Recognition.Tests
{
    public class CrnnTextRecognizerTests
    {
        private readonly CrnnTextRecognizer objectToTest;

        private readonly ColorBoundaryFinder colorBoundaryFinder;

        public CrnnTextRecognizerTests()
        {
            this.objectToTest = new CrnnTextRecognizer();
            this.colorBoundaryFinder = new ColorBoundaryFinder();
        }

        [Theory]
        [InlineData("assets/data/31.png", "31")]
        [InlineData("assets/data/batrider.png", "batrider")]
        [InlineData("assets/data/zero.png", "0")]
        [InlineData("assets/data/30.png", "30")]
        [InlineData("assets/data/660.png", "660")]
        public void Check_Text_Recognition(string fileName, string expected)
        {
            // Arrange
            using var image = new Mat(fileName, loadType: ImreadModes.Grayscale);
            var box = new PointF[] { new PointF(0, image.Rows), new Point(0, 0), new PointF(image.Cols, 0), new PointF(image.Cols, image.Rows) };

            // Act
            var actual = this.objectToTest.Recognize(image, box);

            // Assert
            actual.Should().Be(expected);
        }

        [Theory]
        [InlineData("assets/data/batrider.png", "batrider")]
        [InlineData("assets/data/zero.png", "0")]
        [InlineData("assets/data/30.png", "30")]
        [InlineData("assets/data/660.png", "660")]
        public void Check_Text_Recognition_Bitmap(string fileName, string expected)
        {
            // Skip for linux env
            if (!OperatingSystem.IsWindows())
            {
                return;
            }

            // Arrange
            using var image = (Bitmap)Bitmap.FromFile(fileName);
            var box = new PointF[] { new PointF(0, image.Height), new Point(0, 0), new PointF(image.Width, 0), new PointF(image.Width, image.Height) };

            // Act
            var actual = this.objectToTest.Recognize(image, box);

            // Assert
            actual.Should().Be(expected);
        }

        [Theory]
        [InlineData("assets/data/dark-seer.png", "darkseer")]
        public void Check_Text_Recognition_Ingore_Spaces(string fileName, string expected)
        {
            // Arrange
            using var image = new Mat(fileName, loadType: ImreadModes.Grayscale);
            var box = new PointF[] { new PointF(0, image.Rows), new Point(0, 0), new PointF(image.Cols, 0), new PointF(image.Cols, image.Rows) };

            // Act
            var actual = this.objectToTest.Recognize(image, box);

            // Assert
            actual.Should().Be(expected);
        }

        [Theory]
        [InlineData("assets/data/31.png", "31")]
        public void Check_Text_Recognition_With_Finder(string fileName, string expected)
        {
            // Skip for linux env
            if (!OperatingSystem.IsWindows())
            {
                return;
            }

            // Arrange
            using var bitmap = (Bitmap)Bitmap.FromFile(fileName);
            var rect = this.colorBoundaryFinder.FindBoundary(bitmap, 136, 255, 3);
            
            using var cuttedBitmap = new Bitmap((int)rect.Width, (int)rect.Height);
            using (Graphics g = Graphics.FromImage(cuttedBitmap))
            {
                g.DrawImage(bitmap, new Rectangle(0, 0, (int)rect.Width, (int)rect.Height), new Rectangle((int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height), GraphicsUnit.Pixel);
            }

            var box = new PointF[] { new PointF(0, cuttedBitmap.Height), new Point(0, 0), new PointF(cuttedBitmap.Width, 0), new PointF(cuttedBitmap.Width, cuttedBitmap.Height) };

            // Act
            var actual = this.objectToTest.Recognize(cuttedBitmap, box);

            // Assert
            actual.Should().Be(expected);
#if DEBUG
            
            using (var maskedBitmap = new Bitmap(bitmap))
            {
                using (var g = Graphics.FromImage(maskedBitmap))
                {
                    g.DrawRectangle(new Pen(new SolidBrush(Color.Red)), new Rectangle((int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height));
                }

                var filenameMaskedImage = "assets/tmp/31_recognition_outset.png";
                if (!Directory.Exists(Path.GetDirectoryName(filenameMaskedImage)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(filenameMaskedImage));
                }

                maskedBitmap.Save(filenameMaskedImage);
            }
#endif
        }
    }
}
