using System;
using System.Threading;
using System.Threading.Tasks;
using System.Threading.Tasks.Schedulers;
using Moq;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Parameters
{
    [TestFixture]
    public class GridParameterSelectionTests
    {
        private GridParameterSelection instance;

        private Mock<ITrainingModel> training;

        private GridSearchParameters parameters;

        private Problem problem;

        private TaskFactory taskFactory;

        private ManualResetEvent resetEvent;

        [SetUp]
        public void Setup()
        {
            training = new Mock<ITrainingModel>();
            taskFactory = new TaskFactory(new LimitedConcurrencyLevelTaskScheduler(2));
            parameters = new GridSearchParameters(5, new double[] { 1, 2, 3, 4 }, new double[] { 1, 2, 3, 4 }, new Parameter());
            instance = new GridParameterSelection(taskFactory, training.Object, parameters);

            var dataSet = ArffDataSet.CreateSimple("Test");
            dataSet.Header.RegisterNominalClass("One", "Two", "Three");
            dataSet.UseTotal = true;
            var one = dataSet.AddDocument();
            one.Class.Value = "One";
            one.AddRecord("Good");
            problem = dataSet.GetProblem();
            resetEvent = new ManualResetEvent(false);
        }

        [Test]
        public void Construct()
        {
            Assert.Throws<ArgumentNullException>(() => new GridParameterSelection(null, training.Object, parameters));
            Assert.Throws<ArgumentNullException>(() => new GridParameterSelection(taskFactory, null, parameters));
            Assert.Throws<ArgumentNullException>(() => new GridParameterSelection(taskFactory, training.Object, null));
            Assert.IsNotNull(instance);
        }

        [Test]
        public void Find()
        {
            training.Setup(item => item.PerformCrossValidation(It.IsAny<Problem>(), It.IsAny<Parameter>(), 5))
                .Callback(() => resetEvent.WaitOne());

            var result = instance.Find(problem, CancellationToken.None);
            Thread.Sleep(500);
            Assert.IsFalse(result.IsCompleted);
            resetEvent.Set();

            Thread.Sleep(500);
            Assert.IsTrue(result.IsCompleted);
        }

        [Test]
        public void FindCancel()
        {
            training.Setup(item => item.PerformCrossValidation(It.IsAny<Problem>(), It.IsAny<Parameter>(), 5))
                .Callback(() => resetEvent.WaitOne());

            CancellationTokenSource source = new CancellationTokenSource();
            var result = instance.Find(problem, source.Token);
            Thread.Sleep(500);
            Assert.IsFalse(result.IsCompleted);
            source.Cancel();
            resetEvent.Set();
            Thread.Sleep(500);
            Assert.IsTrue(result.IsCompleted);
        }

        [Test]
        public void FindError()
        {
            training.Setup(item => item.PerformCrossValidation(It.IsAny<Problem>(), It.IsAny<Parameter>(), 5))
                    .Throws<ArgumentNullException>();

            var result = instance.Find(problem, CancellationToken.None);
            Thread.Sleep(500);
            Assert.IsTrue(result.IsCompleted);
            Assert.IsTrue(result.IsFaulted);
        }
    }
}
