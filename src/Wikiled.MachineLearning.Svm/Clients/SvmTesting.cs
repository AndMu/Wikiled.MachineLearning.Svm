using System;
using System.IO;
using System.Linq;
using NLog;
using Wikiled.Arff.Persistence;
using Wikiled.Arff.Persistence.Headers;
using Wikiled.Common.Arguments;
using Wikiled.Common.Extensions;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public class SvmTesting : ISvmTesting
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private readonly Model trainedModel;

        private readonly IProblemFactory problemFactory;

        /// <summary>
        /// Construtor
        /// </summary>
        /// <param name="trainedModel">Trained model</param>
        public SvmTesting(Model trainedModel, IProblemFactory problemFactory)
        {
            Guard.NotNull(() => trainedModel, trainedModel);
            Guard.NotNull(() => problemFactory, problemFactory);
            this.trainedModel = trainedModel;
            this.problemFactory = problemFactory;
        }

        public PredictionResult Classify(IArffDataSet testDataSet)
        {
            Guard.NotNull(() => testDataSet, testDataSet);
            log.Debug("Classify");
            var result = Test(testDataSet);
            var docs = testDataSet.Documents.ToArray();
            for (int i = 0; i < result.Classes.Length; i++)
            {
                var review = docs[i];
                var classValue = ((IClassHeader)review.Class.Header).GetValueByClassId(result.Classes[i].Actual);
                review.Class.Value = classValue;
            }

            return result;
        }

        public double Test(IArffDataSet testingSet, string path)
        {
            Guard.NotNull(() => testingSet, testingSet);
            log.Debug("Test");
            path.EnsureDirectoryExistence();
            var result = Test(testingSet);
            string[] files = Directory.GetFiles(path, "training.*");
            foreach (var file in files)
            {
                if (file.IndexOf("training.model", StringComparison.OrdinalIgnoreCase) == -1)
                {
                    File.Delete(file);
                }
            }

            return result.CorrectProbability;
        }


        public PredictionResult Test(IArffDataSet testingSet)
        {
            Guard.NotNull(() => testingSet, testingSet);
            var problemSource = problemFactory.Construct(testingSet);
            return Prediction.Predict(problemSource.GetProblem(), trainedModel, false);
        }
    }
}
