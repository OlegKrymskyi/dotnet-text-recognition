using Microsoft.ML.Data;

namespace Ok.TextRecognition.Detection.Models
{
    internal class CraftInputModel
    {
        [LoadColumn(0)]
        [ColumnName("image")]
        [VectorType(1, 1184, 1280, 3)]
        public float[] Image;
    }
}
