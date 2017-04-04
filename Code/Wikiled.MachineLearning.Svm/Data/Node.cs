using System;

namespace Wikiled.MachineLearning.Svm.Data
{
    /// <summary>
    /// Encapsulates a node in a Problem vector, with an index and a value (for more efficient representation
    /// of sparse data.
    /// </summary>
	[Serializable]
	public class Node : IComparable<Node>, ICloneable
	{
        /// <summary>
        /// DefaultParallel Constructor.
        /// </summary>
        public Node()
        {
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="index">The index of the value.</param>
        /// <param name="value">The value to store.</param>
        public Node(int index, double value)
        {
            Index = index;
            Value = value;
        }

        /// <summary>
        /// Index of this Node.
        /// </summary>
        public int Index { get; set; }
        
        /// <summary>
        /// Value at Index.
        /// </summary>
        public double Value { get; set; }

        /// <summary>
        /// String representation of this Node as {index}:{value}.
        /// </summary>
        /// <returns>{index}:{value}</returns>
        public override string ToString()
        {
            return $"{Index}:{Value}";
        }

        public object Clone()
        {
            return new Node(Index, Value);
        }

        /// <summary>
        /// Compares this node with another.
        /// </summary>
        /// <param name="other">The node to compare to</param>
        /// <returns>A positive number if this node is greater, a negative number if it is less than, or 0 if equal</returns>
        public int CompareTo(Node other)
        {
            return Index.CompareTo(other.Index);
        }
    }
}