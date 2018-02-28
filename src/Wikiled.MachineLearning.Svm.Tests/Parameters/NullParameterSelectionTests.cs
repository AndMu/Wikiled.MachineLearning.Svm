using System;
using System.Threading;
using System.Threading.Tasks;
using Moq;
using NUnit.Framework;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Tests.Parameters
{
    [TestFixture]
    public class NullParameterSelectionTests
    {
        [Test]
        public async Task Construct()
        {
            Mock<ITrainingModel> model = new Mock<ITrainingModel>();
            Parameter parameter = new Parameter();
            Assert.Throws<ArgumentNullException>(() => new NullParameterSelection(null, model.Object));
            Assert.Throws<ArgumentNullException>(() => new NullParameterSelection(parameter, null));
            
            var instance = new NullParameterSelection(parameter, model.Object);
            var result = await instance.Find(new Problem(), CancellationToken.None).ConfigureAwait(false);
            Assert.AreSame(parameter, result);
        }
    }
}
