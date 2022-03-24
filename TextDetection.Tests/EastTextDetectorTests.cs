using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.CvEnum;
using FluentAssertions;
using Newtonsoft.Json;
using TextDetection.Models;
using Xunit;

namespace TextDetection.Tests
{
    public class EastTextDetectorTests
    {
        private readonly EastTextDetector objectToTest;

        public EastTextDetectorTests()
        {
            this.objectToTest = new EastTextDetector();
        }

        [Theory]
        [InlineData("text-free.png", "text-free.json")]
        [InlineData("hawaii.jpg", "hawaii.json")]
        [InlineData("some-text.png", "some-text.json")]
        public void Check_Text_Detection(string imageFile, string expectedDataFile)
        {
            // Arrange
            using var image = new Mat($"assets/images/{imageFile}", loadType: ImreadModes.Color);

            // Act
            var actual = this.objectToTest.DetectTexts(image);

            // Assert
            var expectedDataJson = File.ReadAllText($"assets/data/{expectedDataFile}");
            var expected = JsonConvert.DeserializeObject<TextDetectionResultModel>(expectedDataJson);
            actual.Should().BeEquivalentTo(expected);
        }
    }
}
