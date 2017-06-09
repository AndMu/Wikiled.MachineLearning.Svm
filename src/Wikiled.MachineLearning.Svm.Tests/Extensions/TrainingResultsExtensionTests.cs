using System;
using System.IO;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Extensions
{
    [TestFixture]
    public class TrainingResultsExtensionTests
    {
        private TrainingResults instance;

        [SetUp]
        public void Setup()
        {
            var dataSet = ArffDataSet.CreateSimple("Test");
            var model = new Model();
            model.NumberOfClasses = 2;
            model.ClassLabels = null;
            model.NumberOfSVPerClass = null;
            model.PairwiseProbabilityA = null;
            model.PairwiseProbabilityB = null;
            model.SupportVectorCoefficients = new double[1][];
            model.Rho = new double[1];
            model.Rho[0] = 0;
            model.Parameter = new Parameter();
            instance = new TrainingResults(model, TrainingHeader.CreateDefault(), dataSet);
        }

        [Test]
        public void SaveLoadArguments()
        {
            Assert.Throws<ArgumentNullException>(() => TrainingResultsExtension.Load(null));
            Assert.Throws<ArgumentOutOfRangeException>(() => TrainingResultsExtension.Load("xxx"));
            Assert.Throws<ArgumentNullException>(() => TrainingResultsExtension.Save(null, "xxx"));
            Assert.Throws<ArgumentNullException>(() => TrainingResultsExtension.Save(instance, null));
        }

        [Test]
        public void SaveLoad()
        {
            var folder = Path.Combine(TestContext.CurrentContext.TestDirectory, "instance");
            instance.Save(folder);
            var result = TrainingResultsExtension.Load(folder);
            Assert.IsNotNull(result.Header);
            Assert.IsNotNull(result.Model);
            Assert.IsNotNull(result.DataSet);
        }
    }
}
