using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Xml.Linq;
using NLog;
using Wikiled.Arff.Persistence;
using Wikiled.Common.Arguments;
using Wikiled.Common.Extensions;
using Wikiled.Common.Serialization;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public static class TrainingResultsExtension
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private static string modelFile = "model.dat";

        private static string arffFile = "result.arff";

        private static string headerFile = "header.xml";

        public static TrainingResults Load(string path)
        {
            Guard.NotNullOrEmpty(() => path, path);
            log.Debug("Load: {0}", path);

            if (File.Exists(path))
            {
                return LoadCompressed(path);
            }

            return LoadNormal(path);
        }

        public static void Save(this TrainingResults result, string path)
        {
            Guard.NotNull(() => result, result);
            Guard.NotNullOrEmpty(() => path, path);
            log.Debug("Save: {0}", path);
            path.EnsureDirectoryExistence();
            result.Header.AverageVectorSize = result.DataSet.Documents.Average(item => item.Count);
            if (result.DataSet.TotalDocuments > 0)
            {
                result.Header.AverageVectorSize = result.DataSet.Documents.Average(item => item.Count);
            }

            using (FileStream stream = new FileStream(Path.Combine(path, arffFile), FileMode.Create))
            {
                SaveArff(result.DataSet, stream);
            }

            using (FileStream stream = new FileStream(Path.Combine(path, modelFile), FileMode.Create))
            {
                result.Model.Write(stream);
            }
        }

        public static void SaveCompressed(this TrainingResults result, string path)
        {
            Guard.NotNull(() => result, result);
            Guard.NotNullOrEmpty(() => path, path);
            log.Debug("SaveCompressed: {0}", path);
            if (result.DataSet.TotalDocuments > 0)
            {
                result.Header.AverageVectorSize = result.DataSet.Documents.Average(item => item.Count);
            }

            using (FileStream zipToOpen = new FileStream(path, FileMode.Create))
            {
                using (ZipArchive archive = new ZipArchive(zipToOpen, ZipArchiveMode.Create))
                {
                    ZipArchiveEntry readmeEntry = archive.CreateEntry(headerFile);
                    using (var stream = readmeEntry.Open())
                    {
                        result.Header.XmlSerialize().Save(stream);
                    }

                    ZipArchiveEntry resultEntry = archive.CreateEntry(arffFile);
                    using (var stream = resultEntry.Open())
                    {
                        SaveArff(result.DataSet, stream);
                    }

                    ZipArchiveEntry modelEntry = archive.CreateEntry(modelFile);
                    using (var stream = modelEntry.Open())
                    {
                        result.Model.Write(stream);
                    }
                }
            }
        }

        private static string GetFile(string path, string name)
        {
            var file = Path.Combine(path, name);
            if (!File.Exists(file))
            {
                throw new ArgumentOutOfRangeException(nameof(path), file);
            }

            return file;
        }

        private static TrainingResults LoadCompressed(string path)
        {
            log.Debug("LoadNormal: {0}", path);
            TrainingHeader header = null;
            Model model = null;
            IArffDataSet dataSet = null;
            using (ZipArchive archive = ZipFile.OpenRead(path))
            {
                foreach (ZipArchiveEntry entry in archive.Entries)
                {
                    if (entry.FullName.EndsWith(headerFile, StringComparison.OrdinalIgnoreCase))
                    {
                        header = XDocument.Load(entry.Open()).XmlDeserialize<TrainingHeader>();
                    }
                    else if (entry.FullName.EndsWith(modelFile, StringComparison.OrdinalIgnoreCase))
                    {
                        model = Model.Read(entry.Open());
                    }
                    else if (entry.FullName.EndsWith(arffFile, StringComparison.OrdinalIgnoreCase))
                    {
                        using (StreamReader reader = new StreamReader(entry.Open()))
                        {
                            dataSet = ArffDataSet.LoadSimple(reader);
                        }
                    }
                }
            }

            return new TrainingResults(model, header, dataSet);
        }

        private static TrainingResults LoadNormal(string path)
        {
            log.Debug("LoadNormal: {0}", path);
            if (!Directory.Exists(path))
            {
                throw new ArgumentOutOfRangeException(nameof(path), path);
            }

            var file = GetFile(path, headerFile);
            var header = File.Exists(file) ? XDocument.Load(file).XmlDeserialize<TrainingHeader>() : null;

            var model = Model.Read(GetFile(path, modelFile));
            IArffDataSet arff;
            using (FileStream stream = new FileStream(GetFile(path, arffFile), FileMode.Open))
            {
                using (StreamReader reader = new StreamReader(stream))
                {
                    arff = ArffDataSet.LoadSimple(reader);
                }
            }

            return new TrainingResults(model, header, arff);
        }

        private static void SaveArff(IArffDataSet arff, Stream outStream)
        {
            using (StreamWriter writer = new StreamWriter(outStream))
            {
                arff.Save(writer);
                outStream.Flush();
            }
        }
    }
}
