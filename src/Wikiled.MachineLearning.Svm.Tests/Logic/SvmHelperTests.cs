using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Clients;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Tests.Logic
{
    [TestFixture]
    public class SvmHelperTests
    {
        private IArffDataSet threeClassDataset;

        private IArffDataSet twoClassDataset;

        [SetUp]
        public void Setup()
        {
            threeClassDataset = ArffDataSet.Create<PositivityType>("Test");
            threeClassDataset.UseTotal = true;
            twoClassDataset = ArffDataSet.CreateSimple("Test");
            twoClassDataset.UseTotal = true;
            twoClassDataset.Header.RegisterNominalClass("Positive", "Negative");

            for (int i = 0; i < 20; i++)
            {
                var positive = threeClassDataset.AddDocument();
                positive.Class.Value = PositivityType.Positive;
                positive.AddRecord("Good");

                positive = twoClassDataset.AddDocument();
                positive.Class.Value = "Positive";
                positive.AddRecord("Good");

                var negative = threeClassDataset.AddDocument();
                negative.Class.Value = PositivityType.Negative;
                negative.AddRecord("Bad");

                negative = twoClassDataset.AddDocument();
                negative.Class.Value = "Negative";
                negative.AddRecord("Bad");
            }
        }

        [Test]
        public async Task Classify()
        {
            var problemFactory = new ProblemFactory(threeClassDataset);
            SvmTraining training = new SvmTraining(problemFactory, threeClassDataset);
            var parameters = training.SelectParameters(TrainingHeader.CreateDefault(), CancellationToken.None);
            var results = await training.Train(parameters).ConfigureAwait(false);
            results.Model.Write(Path.Combine(TestContext.CurrentContext.TestDirectory, "classify.dat"));
            var testSet = ArffDataSet.Create<PositivityType>("Test");
            testSet.UseTotal = true;

            var positive = testSet.AddDocument();
            positive.AddRecord("Good");

            var negative = testSet.AddDocument();
            negative.AddRecord("Bad");

            SvmTesting testing = new SvmTesting(results.Model, problemFactory);
            testing.Classify(testSet);
            Assert.AreEqual(PositivityType.Positive, positive.Class.Value);
            Assert.AreEqual(PositivityType.Negative, negative.Class.Value);
        }

        [Test]
        public async Task TestMultiClass()
        {
            var dataSet = ArffDataSet.CreateSimple("Test");
            dataSet.Header.RegisterNominalClass("One", "Two", "Three");
            dataSet.UseTotal = true;
            for (int i = 0; i < 20; i++)
            {
                var one = dataSet.AddDocument();
                one.Class.Value = "One";
                one.AddRecord("Good");

                var two = dataSet.AddDocument();
                two.Class.Value = "Two";
                two.AddRecord("Bad");

                var three = dataSet.AddDocument();
                three.Class.Value = "Three";
                three.AddRecord("Some");
            }

            var problemFactory = new ProblemFactory(dataSet);
            SvmTraining training = new SvmTraining(problemFactory, dataSet);
            var parameters = training.SelectParameters(TrainingHeader.CreateDefault(), CancellationToken.None);
            var results = await training.Train(parameters).ConfigureAwait(false);

            var file = Path.Combine(TestContext.CurrentContext.TestDirectory, "data.arff");
            dataSet.Save(file);
            results.Model.Write(Path.Combine(TestContext.CurrentContext.TestDirectory, "label.dat"));
            var testFile = ArffDataSet.LoadSimple(file);

            SvmTesting testing = new SvmTesting(results.Model, problemFactory);
            var result = testing.Test(testFile, Path.Combine(TestContext.CurrentContext.TestDirectory, "."));
            Assert.AreEqual(1, result);
        }

        [Test]
        public async Task TestTwoClass()
        {
            var problemFactory = new ProblemFactory(twoClassDataset);
            SvmTraining training = new SvmTraining(problemFactory, twoClassDataset);
            var parameters = training.SelectParameters(TrainingHeader.CreateDefault(), CancellationToken.None);
            var results = await training.Train(parameters).ConfigureAwait(false);
            var file = Path.Combine(TestContext.CurrentContext.TestDirectory, "data.arff");
            threeClassDataset.Save(file);
            var testFile = ArffDataSet.LoadSimple(file);
            SvmTesting testing = new SvmTesting(results.Model, problemFactory);
            var result = testing.Test(testFile, Path.Combine(TestContext.CurrentContext.TestDirectory, "."));
            Assert.AreEqual(1, result);
        }
    }
}
