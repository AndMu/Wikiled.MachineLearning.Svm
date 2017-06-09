using System.Threading;
using System.Threading.Tasks;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public interface ISvmTrain
    {
        Task<TrainingResults> Train(TrainingHeader header, CancellationToken token);
    }
}