using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NLog;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Acceptance
{
    [TestFixture]
    public class TestSvm
    {
        private Parameter parameters;

        private TrainingModel model;

        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        [SetUp]
        public void Setup()
        {
            var header = TrainingHeader.CreateDefault();
            parameters = new Parameter();
            parameters.KernelType = header.Kernel;
            parameters.CacheSize = 200;
            parameters.SvmType = header.SvmType;
            parameters.Probability = false;
            model = new TrainingModel();
        }

        [Test]
        public void Test()
        {
            log.Info("Test");
            var problem = LoadData();
            parameters.Weights = WeightCalculation.GetWeights(problem.Y);

            double test = model.PerformCrossValidation(problem, parameters, 5);
            Assert.AreEqual(0.88, Math.Round(test, 2));
        }

        [Test]
        public async Task Cancel()
        {
            log.Info("Cancel");
            var problem = LoadData();
            parameters.Weights = WeightCalculation.GetWeights(problem.Y);

            CancellationTokenSource source = new CancellationTokenSource();
            parameters.Token = source.Token;
            var task = Task.Run(() => model.PerformCrossValidation(problem, parameters, 5));
            await Task.Delay(5000);
            source.Cancel();
            await Task.Delay(5000);
            Assert.IsTrue(task.IsFaulted);
        }

        private Problem LoadData()
        {
            var file = Path.Combine(TestContext.CurrentContext.TestDirectory, @".\Data\data.arff");
            var arff = ArffDataSet.Load<PositivityType>(file);
            return arff.GetProblem();
        }
    }
}
