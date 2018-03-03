using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Wikiled.Arff.Persistence;
using Wikiled.Arff.Persistence.Headers;
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

        public static IArffDataSet CreateTestDataset(this IArffDataSet baseDataSet)
        {
            return ArffDataSet.CreateFixed((IHeadersWordsHandling)baseDataSet.Header.Clone(), "Test");
        }

        public static void FullSave(this IArffDataSet data, string path, TrainingHeader header = null)
        {
            Guard.NotNull(() => data, data);
            Guard.NotNullOrEmpty(() => path, path);
            Guard.IsValid(() => data, data, item => item.Documents.Any(), "Reviews should be at least one");
            path.EnsureDirectoryExistence();
            data.Save(Path.Combine(path, "data.arff"));
            if (header != null)
            {
                if (data.TotalDocuments > 0)
                {
                    header.AverageVectorSize = data.Documents.Average(item => item.Count);
                }
                
                header.XmlSerialize().Save(Path.Combine(path, "header.xml"));
            }
        }
    }
}
