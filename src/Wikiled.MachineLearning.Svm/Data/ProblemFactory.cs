using Wikiled.Arff.Extensions;
using Wikiled.Arff.Persistence;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Data
{
    public class ProblemFactory : IProblemFactory
    {
        private readonly IArffDataSet baseDataSet;

        private IRangeTransform transform;

        public ProblemFactory(IArffDataSet baseDataSet)
        {
            this.baseDataSet = baseDataSet;
        }

        public IProblemFactory WithRangeScaling()
        {
            var original = new ProblemSource(baseDataSet).GetProblem();
            transform = RangeTransform.Compute(original);
            return this;
        }

        public IProblemFactory WithGaussianScaling()
        {
            var original = new ProblemSource(baseDataSet).GetProblem();
            transform = GaussianTransform.Compute(original);
            return this;
        }

        public IProblemSource Construct(IArffDataSet currentDataSet)
        {
            Guard.NotNull(() => currentDataSet, currentDataSet);
            var dataSet = currentDataSet.CopyDataSet(baseDataSet.Header, "Test");
            return new ProblemSource(dataSet) { Transform = transform };
        }
    }
}
