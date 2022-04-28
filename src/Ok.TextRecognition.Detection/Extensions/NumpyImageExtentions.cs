using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Numpy;
using Ok.TextRecognition.Detection.Models;

namespace Ok.TextRecognition.Detection
{
    public static class NumpyImageExtentions
    {
        public static ImageResizeOutputModel ResizeAspectRatio(this NDarray img, float square_size, Inter interpolation, float mag_ratio = 1)
        {
            var height = img.shape[0];
            var width = img.shape[1];
            var channel = img.shape[2];

            // magnify image size
            var target_size = mag_ratio * Math.Max(height, width);

            // set original image size
            if (target_size > square_size)
            {
                target_size = square_size;
            }

            var ratio = target_size / (float)Math.Max(height, width);

            var target_h = (int)(height * ratio);
            var target_w = (int)(width * ratio);

            using var imgMat = new Mat();
            CvInvoke.Resize(img.ToMatImage<byte>(), imgMat, new Size(target_w, target_h), interpolation: interpolation);

            //# make canvas and paste image
            var target_h32 = target_h;
            var target_w32 = target_w;

            if (target_h % 32 != 0)
            {
                target_h32 = target_h + (32 - target_h % 32);
            }

            if (target_w % 32 != 0)
            {
                target_w32 = target_w + (32 - target_w % 32);
            }

            var resized = imgMat.ToImageNDarray<byte>(target_w32, target_h32, channel);

            return new ImageResizeOutputModel
            {
                Image = resized,
                Ratio = ratio
            };
        }

        //public static Tuple<NDarray, float> get_rotated_box(PointF[] points)
        //{
        //    //Obtain the parameters of a rotated box.
        //    //Returns:
        //    //    The vertices of the rotated box in top-left,
        //    //    top-right, bottom-right, bottom-left order along
        //    //    with the angle of rotation about the bottom left corner.
        //    NDarray pts = null;
        //    try
        //    {
        //        var mp = GeometryFactory.MultiPoint();
        //        foreach (var point in points)
        //        {
        //            mp = mp.Point(point.X, point.Y);
        //        }
                
        //        //mp.Build().Geometries
        //        //mp = geometry.MultiPoint(points = points)
        //        //pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[":-1"];  # noqa: E501
        //    }
        //    catch
        //    {
        //        // There weren't enough points for the minimum rotated rectangle function
        //        pts = points.FromPointsArray();
        //    }

        //    // The code below is taken from
        //    // https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

        //    // sort the points based on their x-coordinates
        //    var xSorted = pts[$"{np.argsort(pts[":,0"])},:"];

        //    // grab the left-most and right-most points from the sorted
        //    // x-roodinate points
        //    var leftMost = xSorted[":2,:"];
        //    var rightMost = xSorted["2:,:"];

        //    // now, sort the left-most coordinates according to their
        //    // y-coordinates so we can grab the top-left and bottom-left
        //    // points, respectively
        //    leftMost = leftMost[$"{np.argsort(leftMost[":,1"])},:"];
        //    var tl = leftMost[0];
        //    var bl = leftMost[1];

        //    // now that we have the top-left coordinate, use it as an
        //    // anchor to calculate the Euclidean distance between the
        //    // top-left and right-most points; by the Pythagorean
        //    // theorem, the point with the largest distance will be
        //    // our bottom-right point
        //    var D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0];
        //    rightMost = rightMost[$"{np.argsort(D)["::- 1"]},:"];
        //    var br = rightMost[0];
        //    var tr = rightMost[0];

        //    // return the coordinates in top-left, top-right,
        //    // bottom-right, and bottom-left order
        //    pts = np.array(new List<NDarray>() { tl, tr, br, bl }, dtype: np.float32);

        //    var rotation = (float)np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]));
        //    return new Tuple<NDarray, float>(pts, rotation);
        //}

   //     public static NDarray WarpBox(NDarray image, PointF[] box, int? target_height = null, int? target_width = null, int margin = 0, bool return_transform = false,
   // bool skip_rotate = false)
   //     {
   //         //Warp a boxed region in an image given by a set of four points into
   //         //a rectangle with a specified width and height.Useful for taking crops
   //         //of distorted or rotated text.
   //         //Args:
   //         //    image: The image from which to take the box
   //         //    box: A list of four points starting in the top left
   //         //        corner and moving clockwise.
   //         //    target_height: The height of the output rectangle
   //         //    target_width: The width of the output rectangle
   //         //    return_transform: Whether to return the transformation
   //         //        matrix with the image.
   //         MCvScalar cval;
   //         if (image.shape.Dimensions.Length == 3)
   //         {
   //             cval = new MCvScalar(0, 0, 0);
   //         }

   //         if (!skip_rotate)
   //         {
   //             box, _ = get_rotated_box(box)
   //         }
   // w, h = get_rotated_width_height(box)
   // assert(target_width is None and target_height is None) or(
   //    target_width is not None and target_height is not None
   //), "Either both or neither of target width and height must be provided."
   // if target_width is None and target_height is None:
   //     target_width = w
   //     target_height = h
   // scale = min(target_width / w, target_height / h)
   // M = cv2.getPerspectiveTransform(
   //     src = box,
   //     dst = np.array(
   //         [
   //             [margin, margin],
   //             [scale * w - margin, margin],
   //             [scale * w - margin, scale * h - margin],
   //             [margin, scale * h - margin],
   //         ]
   //     ).astype("float32"),
   // )
   // crop = cv2.warpPerspective(image, M, dsize = (int(scale * w), int(scale * h)))
   // target_shape = (
   //     (target_height, target_width, 3)
   //     if len(image.shape) == 3
   //     else (target_height, target_width)
   // )
   // full = (np.zeros(target_shape) + cval).astype("uint8")
   // full[: crop.shape[0], : crop.shape[1]] = crop
   // if return_transform:
   //     return full, M
   // return full
   // }
    }
}
