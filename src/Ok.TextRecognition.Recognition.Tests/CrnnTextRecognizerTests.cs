using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using FluentAssertions;
using Xunit;

namespace Ok.TextRecognition.Recognition.Tests
{
    public class CrnnTextRecognizerTests
    {
        private readonly CrnnTextRecognizer objectToTest;

        public CrnnTextRecognizerTests()
        {
            this.objectToTest = new CrnnTextRecognizer();
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
    }
}
