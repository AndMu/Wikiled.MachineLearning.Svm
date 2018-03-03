using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public interface ISvmTesting
    {
        PredictionResult Classify(IArffDataSet testDataSet);

        double Test(IArffDataSet testingSet, string path);

        PredictionResult Test(IArffDataSet testingSet);
    }
}