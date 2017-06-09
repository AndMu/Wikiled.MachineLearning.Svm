using System;
using NUnit.Framework;
using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Tests.Logic
{
    [TestFixture]
    public class TrainingResultsTests
    {
        [Test]
        public void Construct()
        {
            var arff = ArffDataSet.CreateSimple("Test");
            var header = TrainingHeader.CreateDefault();
            var model = new Model();

            Assert.Throws<ArgumentNullException>(() => new TrainingResults(null, header, arff));
            Assert.Throws<ArgumentNullException>(() => new TrainingResults(model, null, arff));
            Assert.Throws<ArgumentNullException>(() => new TrainingResults(model, header, null));
            var instance = new TrainingResults(model, header, arff);
            Assert.IsNotNull(instance.Header);
            Assert.IsNotNull(instance.Model);
        }
    }
}
