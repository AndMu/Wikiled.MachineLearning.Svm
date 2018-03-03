using Wikiled.Arff.Persistence;

namespace Wikiled.MachineLearning.Svm.Data
{
    public interface IProblemFactory
    {
        IProblemFactory WithRangeScaling();

        IProblemFactory WithGaussianScaling();

        IProblemSource Construct(IArffDataSet arff);
    }
}
