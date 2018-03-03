using Wikiled.Arff.Persistence;
using Wikiled.Arff.Persistence.Headers;
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
            var dataSet = ArffDataSet.CreateFixed((IHeadersWordsHandling)baseDataSet.Header.Clone(), "Test");
            foreach (var review in currentDataSet.Documents)
            {
                if (review.Count == 0)
                {
                    continue;
                }

                var newReview = dataSet.AddDocument();
                foreach (var word in review.GetRecords())
                {
                    var addedWord = newReview.AddRecord(word.Header);
                    if (addedWord == null)
                    {
                        continue;
                    }

                    addedWord.Value = word.Value;
                }

                newReview.Class.Value = review.Class.Value;
            }

            return new ProblemSource(dataSet) { Transform = transform };
        }
    }
}
