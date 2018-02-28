using System;
using System.IO;
using System.Linq;
using NLog;
using Wikiled.Arff.Persistence;
using Wikiled.Arff.Persistence.Headers;
using Wikiled.Common.Arguments;
using Wikiled.Common.Extensions;
using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public class SvmTesting : ISvmTesting
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private readonly IArffDataSet trainingVectorSpace;

        private readonly Model trainedModel;

        /// <summary>
        /// Construtor
        /// </summary>
        /// <param name="trainingVectorSpace">Required to know feature space</param>
        /// <param name="trainedModel">Trained model</param>
        public SvmTesting(IArffDataSet trainingVectorSpace, Model trainedModel)
        {
            Guard.NotNull(() => trainingVectorSpace, trainingVectorSpace);
            Guard.NotNull(() => trainedModel, trainedModel);
            this.trainingVectorSpace = trainingVectorSpace;
            this.trainedModel = trainedModel;
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

            trainingVectorSpace.Save(Path.Combine(path, $"testing_data_{result.CorrectProbability}.arff"));
            return result.CorrectProbability;
        }

        public IArffDataSet CreateTestDataset()
        {
            return ArffDataSet.CreateFixed((IHeadersWordsHandling)trainingVectorSpace.Header.Clone(), "Test");
        }

        public PredictionResult Test(IArffDataSet testingSet)
        {
            Guard.NotNull(() => testingSet, testingSet);
            var dataSet = CreateTestDataset();
            foreach (var review in testingSet.Documents)
            {
                if (review.Count == 0)
                {
                    continue;
                }

                var newReview = dataSet.AddDocument();
                foreach (var word in review.GetRecords())
                {
                    var addedWord = newReview.Resolve(word.Header);
                    if (addedWord == null)
                    {
                        continue;
                    }

                    addedWord.Value = word.Value;
                }

                newReview.Class.Value = review.Class.Value;
            }

            Problem testing = dataSet.GetProblem();
            return Prediction.Predict(testing, trainedModel, false);
        }
    }
}
