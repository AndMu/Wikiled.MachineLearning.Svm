using System;
using System.Threading;
using System.Threading.Tasks;
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
            Assert.Throws<ArgumentNullException>(() => new NullParameterSelection(null));
            Parameter parameter = new Parameter();
            var instance = new NullParameterSelection(parameter);
            var result = await instance.Find(new Problem(), CancellationToken.None).ConfigureAwait(false);;
            Assert.AreSame(parameter, result);
        }
    }
}
