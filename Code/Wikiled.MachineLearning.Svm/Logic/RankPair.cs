using System;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    ///     Class encoding a member of a ranked set of labels.
    /// </summary>
    public class RankPair : IComparable<RankPair>
    {
        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="score">Score for this pair</param>
        /// <param name="label">Label associated with the given score</param>
        public RankPair(double score, double label)
        {
            Score = score;
            Label = label;
        }

        /// <summary>
        ///     The Label for this pair.
        /// </summary>
        public double Label { get; }

        /// <summary>
        ///     The score for this pair.
        /// </summary>
        public double Score { get; }

        /// <summary>
        ///     Returns a string representation of this pair.
        /// </summary>
        /// <returns>A string in the for Score:Label</returns>
        public override string ToString()
        {
            return string.Format("{0}:{1}", Score, Label);
        }

        /// <summary>
        ///     Compares this pair to another.  It will end up in a sorted list in decending score order.
        /// </summary>
        /// <param name="other">The pair to compare to</param>
        /// <returns>Whether this should come before or after the argument</returns>
        public int CompareTo(RankPair other)
        {
            return other.Score.CompareTo(Score);
        }
    }
}
