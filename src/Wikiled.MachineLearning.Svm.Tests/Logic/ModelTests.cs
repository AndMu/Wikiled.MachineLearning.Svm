using System.IO;
using NUnit.Framework;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Logic
{
    [TestFixture]
    public class ModelTests
    {
        private Model model;

        [SetUp]
        public void Setup()
        {
            model = new Model();
            model.NumberOfClasses = 2;
            model.ClassLabels = null;
            model.NumberOfSVPerClass = null;
            model.PairwiseProbabilityA = null;
            model.PairwiseProbabilityB = null;
            model.SupportVectorCoefficients = new double[1][];
            model.Rho = new double[1];
            model.Rho[0] = 0;
            model.Parameter = new Parameter();
        }

        [Test]
        public void SaveLoad()
        {
            var file = Path.Combine(TestContext.CurrentContext.TestDirectory, "model.dat");
            model.Parameter.Performance = 12;
            model.Write(file);
            var result = Model.Read(file);
            Assert.NotNull(result);
            Assert.AreEqual(12, result.Parameter.Performance);
        }
    }
}
