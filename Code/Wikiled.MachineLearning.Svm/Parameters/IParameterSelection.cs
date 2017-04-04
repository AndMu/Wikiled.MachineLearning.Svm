using System.Threading;
using System.Threading.Tasks;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    public interface IParameterSelection
    {
        Task<Parameter> Find(Problem problem, CancellationToken token);
    }
}