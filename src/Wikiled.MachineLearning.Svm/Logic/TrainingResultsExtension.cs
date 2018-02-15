using System;
using System.IO;
using System.IO.Compression;
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

            result.Header.XmlSerialize().Save(Path.Combine(path, "header.xml"));
            using (FileStream stream = new FileStream(Path.Combine(path, "result.arff"), FileMode.Create))
            {
                SaveArff(result.DataSet, stream);
            }

            using (FileStream stream = new FileStream(Path.Combine(path, "result.arff"), FileMode.Create))
            {
                result.Model.Write(stream);
            }
        }

        public static void SaveCompressed(this TrainingResults result, string path)
        {
            Guard.NotNull(() => result, result);
            Guard.NotNullOrEmpty(() => path, path);
            log.Debug("SaveCompressed: {0}", path);

            using (FileStream zipToOpen = new FileStream(path, FileMode.Create))
            {
                using (ZipArchive archive = new ZipArchive(zipToOpen, ZipArchiveMode.Create))
                {
                    ZipArchiveEntry readmeEntry = archive.CreateEntry("header.xml");
                    using (var stream = readmeEntry.Open())
                    {
                        result.Header.XmlSerialize().Save(stream);
                    }

                    ZipArchiveEntry resultEntry = archive.CreateEntry("result.arff");
                    using (var stream = resultEntry.Open())
                    {
                        SaveArff(result.DataSet, stream);
                    }

                    ZipArchiveEntry modelEntry = archive.CreateEntry("model.dat");
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
                    if (entry.FullName.EndsWith("header.xml", StringComparison.OrdinalIgnoreCase))
                    {
                        header = XDocument.Load(entry.Open()).XmlDeserialize<TrainingHeader>();
                    }
                    else if (entry.FullName.EndsWith("model.dat", StringComparison.OrdinalIgnoreCase))
                    {
                        model = Model.Read(entry.Open());
                    }
                    else if (entry.FullName.EndsWith("result.arff", StringComparison.OrdinalIgnoreCase))
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

            var file = GetFile(path, "header.xml");
            var header = File.Exists(file) ? XDocument.Load(file).XmlDeserialize<TrainingHeader>() : null;

            var model = Model.Read(GetFile(path, "model.dat"));
            IArffDataSet arff;
            using (FileStream stream = new FileStream(GetFile(path, "result.arff"), FileMode.Open))
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
