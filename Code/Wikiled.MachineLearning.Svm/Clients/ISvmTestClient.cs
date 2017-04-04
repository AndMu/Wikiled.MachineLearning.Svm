using Wikiled.Arff.Persistence;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public interface ISvmTestClient
    {
        IArffDataSet CreateTestDataset();

        void Classify(IArffDataSet testDataSet);

        double Test(IArffDataSet testingSet, string path);

        PredictionResult Test(IArffDataSet testingSet);
    }
}