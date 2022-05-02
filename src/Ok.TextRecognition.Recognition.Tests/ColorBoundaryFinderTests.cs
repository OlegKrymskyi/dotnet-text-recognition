using System;
using System.Drawing;
using System.IO;
using FluentAssertions;
using Xunit;

namespace Ok.TextRecognition.Recognition.Tests
{
    public class ColorBoundaryFinderTests
    {
        private readonly ColorBoundaryFinder objectToTest;

        public ColorBoundaryFinderTests()
        {
            this.objectToTest = new ColorBoundaryFinder();
        }

        [Fact]
        public void Find_Boundaries()
        {
            // Skip for linux env
            if (!OperatingSystem.IsWindows())
            {
                return;
            }

            // Arrange
            using var bitmap = (Bitmap)Bitmap.FromFile("assets/data/31.png");

            // Act
            var actual = this.objectToTest.FindBoundaries(bitmap, (byte)136, (byte)255);

            actual.Should().BeEquivalentTo(new RectangleF[] 
            {
                new RectangleF(30, 13, 4, 11),
                new RectangleF(17, 13, 7, 12)
            });

            // Assert
#if DEBUG
            using (var maskedBitmap = new Bitmap(bitmap))
            {
                using (var g = Graphics.FromImage(maskedBitmap))
                {
                    foreach (var rect in actual)
                    {
                        g.DrawRectangle(new Pen(new SolidBrush(Color.Red)), new Rectangle((int)rect.X, (int)rect.Y, (int)rect.Width, (int)rect.Height));
                    }
                }

                var filenameMaskedImage = "assets/tmp/31_boundaries.png";
                if (!Directory.Exists(Path.GetDirectoryName(filenameMaskedImage)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(filenameMaskedImage));
                }

                maskedBitmap.Save(filenameMaskedImage);
            }
#endif
        }

        [Fact]
        public void Find_Boundary()
        {
            // Skip for linux env
            if (!OperatingSystem.IsWindows())
            {
                return;
            }

            // Arrange
            using var bitmap = (Bitmap)Bitmap.FromFile("assets/data/31.png");

            // Act
            var actual = this.objectToTest.FindBoundary(bitmap, (byte)136, (byte)255);
            actual.X.Should().Be(17);
            actual.Y.Should().Be(13);
            actual.Width.Should().Be(17);
            actual.Height.Should().Be(12);

            // Assert
#if DEBUG
            using (var maskedBitmap = new Bitmap(bitmap))
            {
                using (var g = Graphics.FromImage(maskedBitmap))
                {
                    g.DrawRectangle(new Pen(new SolidBrush(Color.Red)), new Rectangle((int)actual.X, (int)actual.Y, (int)actual.Width, (int)actual.Height));
                }

                var filenameMaskedImage = "assets/tmp/31_boundary.png";
                if (!Directory.Exists(Path.GetDirectoryName(filenameMaskedImage)))
                {
                    Directory.CreateDirectory(Path.GetDirectoryName(filenameMaskedImage));
                }

                maskedBitmap.Save(filenameMaskedImage);
            }
#endif
        }

        [Fact]
        public void Find_Boundary_With_Outset()
        {
            // Skip for linux env
            if (!OperatingSystem.IsWindows())
            {
                return;
            }

            // Arrange
            using var bitmap = (Bitmap)Bitmap.FromFile("assets/data/31.png");

            // Act
            var actual = this.objectToTest.FindBoundary(bitmap, (byte)136, (byte)255, 2);
            actual.X.Should().Be(15);
            actual.Y.Should().Be(11);
            actual.Width.Should().Be(21);
            actual.Height.Should().Be(16);

            // Assert
#if DEBUG
            using (var maskedBitmap = new Bitmap(bitmap))
            {
                using (var g = Graphics.FromImage(maskedBitmap))
                {
                    g.DrawRectangle(new Pen(new SolidBrush(Color.Red)), new Rectangle((int)actual.X, (int)actual.Y, (int)actual.Width, (int)actual.Height));
                }

                var filenameMaskedImage = "assets/tmp/31_boundary_outset.png";
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
