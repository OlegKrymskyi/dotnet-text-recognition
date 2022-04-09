using Microsoft.ML.Data;

namespace TextDetection.Models
{
    internal class CraftOutputModel
    {
        [ColumnName("output")]
        public float[] Output;
    }
}
