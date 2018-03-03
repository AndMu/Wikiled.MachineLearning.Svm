using System.IO;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Tests.Logic
{
    [TestFixture]
    public class GaussianTransformTests
    {
        [Test]
        public void Compute()
        {
            var dataSet = ArffDataSet.LoadSimple(Path.Combine(TestContext.CurrentContext.TestDirectory, "Data", @"problem.arff"));
            var problem = new ProblemSource(dataSet).GetProblem();
            var transform = GaussianTransform.Compute(problem);
            var result = transform.Scale(problem);
            Assert.IsNotNull(result);
        }
    }
}