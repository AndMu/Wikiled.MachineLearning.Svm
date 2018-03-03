using System;
using System.Collections.Generic;
using Wikiled.Arff.Persistence;
using Wikiled.Arff.Persistence.Headers;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Data
{
    public class ProblemSource : IProblemSource
    {
        private IArffDataSet arff;

        public ProblemSource(IArffDataSet arff)
        {
            Guard.NotNull(() => arff, arff);
            this.arff = arff;
        }

        public Func<IArffDataRow, bool> Filter { get; set; }

        public IRangeTransform Transform { get; set; }

        public Problem GetProblem()
        {
            var lines = new List<DataLine>();
            foreach (var review in arff.Documents)
            {
                if (Filter != null &&
                    !Filter(review))
                {
                    continue;
                }

                int? classId = review.Class.Value == null ? (int?)null : arff.Header.Class.ReadClassIdValue(review.Class);
                var dataLine = new DataLine(classId);
                review.ProcessLine(dataLine);
                if (dataLine.TotalValues > 0)
                {
                    lines.Add(dataLine);
                }
            }

            var problem = Problem.Read(lines.ToArray());
            return Transform != null ? Transform.Scale(problem) : problem;
        }
    }
}
