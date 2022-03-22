﻿using System;
using System.Drawing;
using System.IO;
using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;

namespace TextRecognition
{
    public class CrnnTextRecognizer
    {
        private readonly TextRecognitionModel model;

        private readonly Size inputSize;

        public CrnnTextRecognizer(string modelFile = "assets/CRNN_VGG_BiLSTM_CTC.onnx", string vocabularyFile = "assets/alphabet_36.txt")
        {
            this.inputSize = new Size(100, 32);
            this.model = new TextRecognitionModel(modelFile);

            this.model.Vocabulary = File.ReadAllText(vocabularyFile).Split(new string[] { Environment.NewLine }, StringSplitOptions.RemoveEmptyEntries);
            this.model.DecodeType = "CTC-greedy";

            this.model.SetInputScale(1.0 / 127.5);
            this.model.SetInputMean(new MCvScalar(127.5, 127.5, 127.5));
            this.model.SetInputSize(this.inputSize);
        }

        public string Recognize(Mat image, PointF[] box)
        {
            PointF[] targetVertices = new PointF[]
            {
                new PointF(0, this.inputSize.Height - 1),
                new PointF(0, 0),
                new PointF(this.inputSize.Width - 1, 0),
                new PointF(this.inputSize.Width - 1, this.inputSize.Height - 1),
            };
            using var rotationMatrix = CvInvoke.GetPerspectiveTransform(box, targetVertices);

            Mat cropped = new Mat();
            CvInvoke.WarpPerspective(image, cropped, rotationMatrix, this.inputSize);

            cropped.Save("cropped.jpg");

            return model.Recognize(cropped);
        }
    }
}
