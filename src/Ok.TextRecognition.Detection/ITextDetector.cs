using System;
using System.Drawing;
using Emgu.CV;
using Ok.TextRecognition.Detection.Models;

namespace Ok.TextRecognition.Detection
{
    public interface ITextDetector: IDisposable
    {
        TextDetectionResultModel DetectTexts(string imageFile);

        TextDetectionResultModel DetectTexts(Bitmap bitmap);

        TextDetectionResultModel DetectTexts(Mat imageMat);
    }
}
