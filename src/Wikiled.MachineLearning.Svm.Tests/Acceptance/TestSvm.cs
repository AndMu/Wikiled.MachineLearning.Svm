using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using NLog;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Acceptance
{
    [TestFixture]
    public class TestSvm
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private Parameter parameters;

        private TrainingModel model;

        [SetUp]
        public void Setup()
        {
            var header = TrainingHeader.CreateDefault();
            parameters = new Parameter();
            parameters.KernelType = header.Kernel;
            parameters.CacheSize = 200;
            parameters.SvmType = header.SvmType;
            parameters.Probability = false;
            model = new TrainingModel(header);
        }

        [Test]
        public void Train()
        {
            log.Info("Test");
            var problem = LoadData();
            parameters.Weights = WeightCalculation.GetWeights(problem.Y);
            model.Train(problem, parameters);
        }

        [TestCase(true, 0.81)]
        [TestCase(false, 0.88)]
        public void PerformCrossValidation(bool scaling, double expected)
        {
            log.Info("Test");
            var problem = LoadData(scaling);
            parameters.Weights = WeightCalculation.GetWeights(problem.Y);

            double test = model.PerformCrossValidation(problem, parameters, 5);
            Assert.AreEqual(expected, Math.Round(test, 2));
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
            await Task.Delay(100).ConfigureAwait(false);
            source.Cancel();
            try
            {
                task.Wait(5000);
            }
            catch
            {
            }

            Assert.IsTrue(task.IsFaulted);
        }

        private Problem LoadData(bool withScaling = false)
        {
            var file = Path.Combine(TestContext.CurrentContext.TestDirectory, @".\Data\data.arff");
            var arff = ArffDataSet.Load<PositivityType>(file);
            IProblemFactory factory = new ProblemFactory(arff);
            if (withScaling)
            {
                factory = factory.WithRangeScaling();
            }

            return factory.Construct(arff).GetProblem();
        }
    }
}
