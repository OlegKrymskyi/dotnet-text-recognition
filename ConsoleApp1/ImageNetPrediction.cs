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
        [ColumnName("273")]
        public float[] textmap;

        [ColumnName("263")]
        public float[] linkmap;
    }
}
