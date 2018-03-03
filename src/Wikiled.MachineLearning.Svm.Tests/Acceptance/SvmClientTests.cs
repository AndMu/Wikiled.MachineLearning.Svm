using System.IO;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Clients;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Tests.Acceptance
{
    [TestFixture]
    public class SvmClientTests
    {

        [TestCase("data_b.arff", "classify.dat", PositivityType.Positive)]
        [TestCase("testing_data_1.arff", "label.dat", "One")]
        public void Classify(string arff, string modelName, object result)
        {
            var file = Path.Combine(TestContext.CurrentContext.TestDirectory, "data", arff);
            var dataSet = result.GetType() == typeof(PositivityType)
                              ? ArffDataSet.Load<PositivityType>(file)
                              : ArffDataSet.LoadSimple(file);
            file = Path.Combine(TestContext.CurrentContext.TestDirectory, "data", modelName);
            var model = Model.Read(file);

            IProblemFactory factory = new ProblemFactory(dataSet);
            var client = new SvmTesting(model, factory);

            var dataHolder = dataSet.CreateTestDataset();
            var review = dataHolder.AddDocument();
            review.AddRecord("Good").Value = 2;
            review.AddRecord("Bad").Value = 1;
            client.Classify(dataHolder);
            Assert.AreEqual(result, review.Class.Value);
        }
    }
}
