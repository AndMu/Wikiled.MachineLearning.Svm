using System.Collections.Generic;
using System.Linq;

namespace Wikiled.MachineLearning.Svm.Data
{
    public class DataLine
    {
        private readonly Dictionary<int, Node> values = new Dictionary<int, Node>();

        public DataLine(double? value)
        {
            Value = value;
            MaxIndex = 0;
        }

        public int MaxIndex { get; private set; }

        public int TotalValues => values.Count;

        public double? Value { get; }

        public Node AddValue(int index, double value)
        {
            MaxIndex = index > MaxIndex ? index : MaxIndex;
            Node existing;
            if (!values.TryGetValue(index, out existing))
            {
                return SetValue(index, value);
            }

            existing.Value += value;
            return existing;
        }

        public Node[] GetVectorX()
        {
            return values.Select(value => value.Value).ToArray();
        }

        public Node SetValue(int index, double value)
        {
            MaxIndex = index > MaxIndex ? index : MaxIndex;
            Node node = new Node(index, value);
            values[index] = node;
            return node;
        }
    }
}
