using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Data
{
    public interface IProblemSource
    {
        Problem GetProblem();
    }
}