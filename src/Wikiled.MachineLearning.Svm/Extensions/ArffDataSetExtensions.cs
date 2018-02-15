using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Wikiled.Arff.Persistence;
using Wikiled.Common.Arguments;
using Wikiled.Common.Extensions;
using Wikiled.Common.Serialization;
using Wikiled.MachineLearning.Svm.Logic;
using DataLine = Wikiled.MachineLearning.Svm.Data.DataLine;

namespace Wikiled.MachineLearning.Svm.Extensions
{
    public static class ArffDataSetExtensions
    {
        public static void SaveSorted(this IArffDataSet arff, string outPath)
        {
            using (var stream = new StreamWriter(outPath))
            {
                stream.WriteLine(arff.ToString());
                foreach (var review in arff.Documents)
                {
                    var text = review.ToString();
                    if (string.IsNullOrEmpty(text))
                    {
                        return;
                    }

                    stream.WriteLine(text);
                }
            }
        }

        public static Problem GetProblem(this IArffDataSet arff, Func<IArffDataRow, bool> filter = null)
        {
            var lines = new List<DataLine>();
            foreach (var review in arff.Documents)
            {
                if (filter != null &&
                    !filter(review))
                {
                    continue;
                }

                int? classId = review.Class.Value == null ? (int?)null : arff.Header.Class.ReadClassIdValue(review.Class);
                var dataLine = new DataLine(classId);
                review.ProcessLine(dataLine);
                if (dataLine.TotalValues > 0)
                {
                    lines.Add(dataLine);
                }
            }

            return Problem.Read(lines.ToArray());
        }

        public static void FullSave(this IArffDataSet data, string path, TrainingHeader header = null)
        {
            Guard.NotNull(() => data, data);
            Guard.NotNullOrEmpty(() => path, path);
            Guard.IsValid(() => data, data, item => item.Documents.Any(), "Reviews should be at least one");
            DirectoryInfoExtensions.EnsureDirectoryExistence(path);
            data.Save(Path.Combine(path, "data.arff"));
            if (header != null)
            {
                header.AverageVectorSize = data.Documents.Average(item => item.Count);
                header.XmlSerialize().Save(Path.Combine(path, "header.xml"));
            }
        }
    }
}
