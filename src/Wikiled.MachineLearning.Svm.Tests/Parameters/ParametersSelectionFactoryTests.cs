using System;
using System.Threading.Tasks;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Parameters
{
    [TestFixture]
    public class ParametersSelectionFactoryTests
    {
        private ParametersSelectionFactory factory;

        private IArffDataSet dataSet;

        [SetUp]
        public void Setup()
        {
            dataSet = ArffDataSet.CreateSimple("Test");
            dataSet.Header.RegisterNominalClass("One");
            var problemFactory = new ProblemFactory(dataSet);
            factory = new ParametersSelectionFactory(Task.Factory, problemFactory);
        }

        [Test]
        public void CreateNull()
        {
            TrainingHeader header = TrainingHeader.CreateDefault();
            Assert.Throws<ArgumentNullException>(() => factory.Create(null, dataSet));
            Assert.Throws<ArgumentNullException>(() => factory.Create(header, null));
            header.GridSelection = false;
            var result = factory.Create(header, ArffDataSet.CreateSimple("Test"));
            Assert.IsInstanceOf<NullParameterSelection>(result);
        }

        [TestCase(KernelType.Linear, 1, 100, 1, 4, false)]
        [TestCase(KernelType.Linear, 1, 1, 1, 4, true)]
        [TestCase(KernelType.RBF, 1, 1, 10, 11, true)]
        public void CreateGrid(KernelType kernel, int instances, int features, int gammas, int c, bool shrink)
        {
            for (int i = 0; i < features; i++)
            {
                dataSet.Header.RegisterNumeric(i.ToString());
            }

            for (int i = 0; i < instances; i++)
            {
                var review = dataSet.AddDocument();
                review.Class.Value = "One";
                review.AddRecord("Record").Value = 0.1;
            }

            TrainingHeader header = TrainingHeader.Create(kernel, SvmType.C_SVC);
            var result = factory.Create(header, dataSet) as GridParameterSelection;
            Assert.AreEqual(3, result.SearchParameters.Folds);
            Assert.AreEqual(gammas, result.SearchParameters.Gamma.Length);
            Assert.AreEqual(c, result.SearchParameters.C.Length);
            Assert.AreEqual(kernel, result.SearchParameters.Default.KernelType);
            Assert.AreEqual(shrink, result.SearchParameters.Default.Shrinking);
            Assert.AreEqual(SvmType.C_SVC, result.SearchParameters.Default.SvmType);
        }
    }
}
