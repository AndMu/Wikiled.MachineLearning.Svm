using System;
using System.Collections.Generic;
using System.Linq;
using Wikiled.Arff.Persistence;
using Wikiled.Arff.Persistence.Headers;
using DataLine = Wikiled.MachineLearning.Svm.Data.DataLine;

namespace Wikiled.MachineLearning.Svm.Extensions
{
    public static class ArffDataRowExtensions
    {
        public static void ProcessLine(this IArffDataRow row, DataLine line)
        {
            var indexes = new Dictionary<int, double>();
            foreach (var wordsData in row.GetRecords())
            {
                if (wordsData.Header is DateHeader)
                {
                    continue;
                }

                int index = row.Owner.Header.GetIndex(wordsData.Header);
                double value = 1;
                if (wordsData.Value != null)
                {
                    value = Convert.ToDouble(wordsData.Value);
                }

                indexes[index] = value;
            }

            foreach (var index in indexes.OrderBy(item => item.Key))
            {
                line.SetValue(index.Key, index.Value);
            }
        }
    }
}
