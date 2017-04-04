using System.Collections.Generic;
using System.Linq;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class PredictionResult
    {
        private readonly List<ClassificationClass> classes = new List<ClassificationClass>();

        private readonly List<int> labels = new List<int>();

        public ClassificationClass[] Classes => classes.ToArray();

        public double CorrectProbability { get; set; }

        public int[] Labels => labels.ToArray();

        public int Total => classes.Count;

        public void AddLabel(int label)
        {
            labels.Add(label);
        }

        public double CalculatePrecision(double clasificationClass)
        {
            double totalPositive = GetTotalTruePositiveClassifications(clasificationClass);
            double totalFalsePositive = GetTotalFalsePositiveClassifications(clasificationClass);
            var value = totalPositive / (totalPositive + totalFalsePositive);
            return value;
        }

        public double CalculateRecall(double clasificationClass)
        {
            double totalPositive = GetTotalTruePositiveClassifications(clasificationClass);
            double totalFalseNegative = GetTotalFalseNegativeClassifications(clasificationClass);

            var value = totalPositive / (totalPositive + totalFalseNegative);
            return value;
        }

        public int GetTotalFalseNegativeClassifications(double clasificationClass)
        {
            return classes.Count(item => item.Actual != item.Target && item.Actual != clasificationClass);
        }

        public int GetTotalFalsePositiveClassifications(double clasificationClass)
        {
            return classes.Count(item => item.Actual != item.Target && item.Actual == clasificationClass);
        }

        public int GetTotalTruePositiveClassifications(double clasificationClass)
        {
            return classes.Count(item => item.Actual == item.Target && item.Actual == clasificationClass);
        }

        public void Set(ClassificationClass item)
        {
            classes.Add(item);
        }
    }
}
