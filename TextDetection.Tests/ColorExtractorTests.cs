using System;
using System.Drawing;
using Xunit;
using FluentAssertions;

namespace TextDetection.Tests
{
    public class ColorExtractorTests
    {
        private readonly ColorExtractor objectToTest;

        public ColorExtractorTests()
        {
            objectToTest = new ColorExtractor();
        }

        [Theory]
        [InlineData("monkey-king.png", 72, 9)]
        public void Extracts_Properly(string filename, int width, int height)
        {
            // Arrange
            using var original = (Bitmap)Bitmap.FromFile($"assets/{filename}");

            // Act
            using var result = objectToTest.ExtractColor(original, Color.FromArgb(131, 139, 114), Color.FromArgb(191, 208, 130));

            // Assert
            result.Width.Should().Be(width);
            result.Height.Should().Be(height);
            result.Save("monkey-king-result.png");
        }
    }
}
