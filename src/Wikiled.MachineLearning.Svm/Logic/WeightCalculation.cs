using System.Collections.Generic;
using System.Linq;
using Wikiled.Common.Arguments;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public static class WeightCalculation
    {
        public static Dictionary<int, double> GetWeights(double[] values)
        {
            Guard.NotNull(() => values, values);
            Guard.IsValid(() => values, values, doubles => doubles.Length > 0, "Array must be non-zero");
            // http://stats.stackexchange.com/questions/24959/a-priori-selection-of-svm-class-weights
            // training samples in class l1 om 1 and l2 -- in class 2, take C1 and C2 such that C1/C2 = l2/l1.
            Dictionary<int, List<double>> classSums = new Dictionary<int, List<double>>();
            foreach (var value in values)
            {
                List<double> records;
                if (!classSums.TryGetValue((int)value, out records))
                {
                    records = new List<double>();
                    classSums[(int)value] = records;
                }

                records.Add(value);
            }

            var minimum = classSums.Select(item => item.Value.Count).Min();
            Dictionary<int, double> weights = new Dictionary<int, double>();
            foreach (var classRecord in classSums)
            {
                if (classRecord.Value.Count == minimum)
                {
                    weights[classRecord.Key] = 1;
                }
                else
                {
                    weights[classRecord.Key] = minimum / (double)classRecord.Value.Count;
                }
            }

            return weights;
        }
    }
}
