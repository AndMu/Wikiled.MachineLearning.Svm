using System.Collections.Generic;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Mathematics;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class PredictionResult
    {
        private readonly List<ClassificationClass> classes = new List<ClassificationClass>();

        private readonly List<int> labels = new List<int>();

        public PredictionResult(Parameter parameter)
        {
            Guard.NotNull(() => parameter, parameter);
            Parameter = parameter;
        }

        public ClassificationClass[] Classes => classes.ToArray();

        public Parameter Parameter { get; }

        public double CorrectProbability { get; set; }

        public int[] Labels => labels.ToArray();

        public int Total => classes.Count;

        public PrecisionRecallCalculator<int> Statistics { get; } = new PrecisionRecallCalculator<int>();

        public void AddLabel(int label)
        {
            labels.Add(label);
        }

        public void Set(ClassificationClass item)
        {
            Statistics.Add(item.Target, item.Actual);
            classes.Add(item);
        }
    }
}
