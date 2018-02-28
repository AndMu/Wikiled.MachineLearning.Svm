using System.Threading;
using System.Threading.Tasks;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public interface ISvmTraining
    {
        IParameterSelection SelectParameters(TrainingHeader header, CancellationToken token);

        Task<TrainingResults> Train(IParameterSelection selection);
    }
}