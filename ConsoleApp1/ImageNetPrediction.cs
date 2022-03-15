using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ConsoleApp1
{
    public class ImageNetPrediction
    {
        [ColumnName("textmap")]
        public float[] textmap;

        [ColumnName("linkmap")]
        public float[] linkmap;
    }
}
