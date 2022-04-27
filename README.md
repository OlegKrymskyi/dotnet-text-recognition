# Text Recognition
Text recognition exercise could be splitted into two part: text detection and text recognition itself.

The detected text could be easily recognized with the one of the trained CRNN networks. Take a look on https://drive.google.com/drive/folders/1cTbQ3nuZG-EKWak6emD_s8_hHXWz7lAr

## Text Detection
### CRAFT: Character-Region Awareness For Text detection
Official Pytorch implementation of CRAFT text detector | [Paper](https://arxiv.org/abs/1904.01941) | [Pretrained Model](https://drive.google.com/open?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ) | [Supplementary](https://youtu.be/HI8MzpY8KMI)

### EAST

## Text Recognition
The TextRecognition project contains classes which allows to recognize text on the given image.

```
// Load the image as an OpenCv matrix. Bitmap could be easily converted to this type. Take a look on OpenCvExtension class.
using var image = new Mat(fileName, loadType: ImreadModes.Grayscale);

// Define the box in which the text is located in. Remember, this class recognize only the text, not detecting. You have to detect it first.
// In this example we are loading image which contains only the text. So, we can ask to recognize text on whole image.
var box = new PointF[] { new PointF(0, image.Rows), new Point(0, 0), new PointF(image.Cols, 0), new PointF(image.Cols, image.Rows) };

// Get the recognized text.
var text = this.objectToTest.Recognize(image, box);
```