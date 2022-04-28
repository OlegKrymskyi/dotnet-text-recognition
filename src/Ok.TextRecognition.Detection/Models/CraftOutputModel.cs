using Microsoft.ML.Data;

namespace Ok.TextRecognition.Detection.Models
{
    internal class CraftOutputModel
    {
        [ColumnName("output")]
        public float[] Output;
    }
}
