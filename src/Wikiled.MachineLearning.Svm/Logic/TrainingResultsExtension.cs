using System;
using System.IO;
using System.Xml.Linq;
using NLog;
using Snappy;
using Wikiled.Arff.Persistence;
using Wikiled.Core.Utility.Arguments;
using Wikiled.Core.Utility.Extensions;
using Wikiled.Core.Utility.Serialization;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public static class TrainingResultsExtension
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        public static TrainingResults Load(string path)
        {
            Guard.NotNullOrEmpty(() => path, path);
            log.Debug("Load: {0}", path);
            if (!Directory.Exists(path))
            {
                throw new ArgumentOutOfRangeException(nameof(path), path);
            }

            var file = GetFile(path, "header.xml");
            var header = File.Exists(file) ? XDocument.Load(file).XmlDeserialize<TrainingHeader>() : null;

            var model = Model.Read(GetFile(path, "model.dat"));
            var bytes = File.ReadAllBytes(GetFile(path, "arff.dat"));
            IArffDataSet arff;
            using (MemoryStream stream = new MemoryStream(SnappyCodec.Uncompress(bytes)))
            {
                using (StreamReader reader = new StreamReader(stream))
                {
                    arff = ArffDataSet.LoadSimple(reader);
                }
            }

            return new TrainingResults(model, header, arff);
        }

        public static void Save(this TrainingResults result, string path)
        {
            Guard.NotNull(() => result, result);
            Guard.NotNullOrEmpty(() => path, path);
            log.Debug("Save: {0}", path);
            path.EnsureDirectoryExistence();
            result.Header.XmlSerialize().Save(Path.Combine(path, "header.xml"));
            SaveArff(result.DataSet, path);
            SaveModel(result.Model, path);
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

        private static void SaveArff(IArffDataSet arff, string path)
        {
            using (MemoryStream stream = new MemoryStream())
            {
                using (StreamWriter writer = new StreamWriter(stream))
                {
                    arff.Save(writer);
                }

                stream.Flush();
                var data = SnappyCodec.Compress(stream.ToArray());
                var file = Path.Combine(path, "arff.dat");
                File.WriteAllBytes(file, data);
            }
        }

        private static void SaveModel(Model model, string path)
        {
            using (FileStream stream = new FileStream(Path.Combine(path, "model.dat"), FileMode.Create))
            {
                model.Write(stream);
            }
        }
    }
}
