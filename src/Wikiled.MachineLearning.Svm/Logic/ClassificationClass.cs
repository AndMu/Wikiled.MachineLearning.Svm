using System.Collections.Generic;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class ClassificationClass
    {
        readonly List<double> values = new List<double>();

        public void Add(double value)
        {
            values.Add(value);
        }

        public double[] Values => values.ToArray();

        public double Target { get; set; }

        public double Actual { get; set; }
    }
}
