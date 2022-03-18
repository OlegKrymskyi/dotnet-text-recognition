using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace ConsoleApp1
{
    public class ImageNetData
    {
        [LoadColumn(0)]
        public string ImagePath;
    }
}
