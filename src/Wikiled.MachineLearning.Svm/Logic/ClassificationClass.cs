using System.Collections.Generic;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class ClassificationClass
    {
        private readonly List<double> values = new List<double>();

        public int Actual { get; set; }

        public int Target { get; set; }

        public double[] Values => values.ToArray();

        public void Add(double value)
        {
            values.Add(value);
        }
    }
}
