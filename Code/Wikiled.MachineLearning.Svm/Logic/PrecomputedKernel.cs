using System;
using System.Collections.Generic;
using Wikiled.MachineLearning.Svm.Parameters;
using Node = Wikiled.MachineLearning.Svm.Data.Node;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    /// Class encapsulating a precomputed kernel, where each position indicates the similarity score for two items in the training data.
    /// </summary>
    [Serializable]
    public class PrecomputedKernel
    {
        private readonly float[,] similarities;

        private readonly int rows;

        private readonly int columns;

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="similarities">The similarity scores between all items in the training data</param>
        public PrecomputedKernel(float[,] similarities)
        {
            this.similarities = similarities;
            rows = this.similarities.GetLength(0);
            columns = this.similarities.GetLength(1);
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="nodes">Nodes for self-similarity analysis</param>
        /// <param name="param">Parameters to use when computing similarities</param>
        public PrecomputedKernel(List<Node[]> nodes, Parameter param)
        {
            rows = nodes.Count;
            columns = rows;
            similarities = new float[rows, columns];
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < r; c++)
                {
                    similarities[r, c] = similarities[c, r];
                }
                similarities[r, r] = 1;
                for (int c = r + 1; c < columns; c++)
                {
                    {
                        similarities[r, c] = (float) Kernel.KernelFunction(nodes[r], nodes[c], param);
                    }
                }
            }
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="rows">Nodes to use as the rows of the matrix</param>
        /// <param name="columns">Nodes to use as the columns of the matrix</param>
        /// <param name="param">Parameters to use when compute similarities</param>
        public PrecomputedKernel(List<Node[]> rows, List<Node[]> columns, Parameter param)
        {
            this.rows = rows.Count;
            this.columns = columns.Count;
            similarities = new float[this.rows,this.columns];
            for (int r = 0; r < this.rows; r++)
            {
                for (int c = 0; c < this.columns; c++)
                {
                    similarities[r, c] = (float) Kernel.KernelFunction(rows[r], columns[c], param);
                }
            }
        }

        /// <summary>
        /// Constructs a <see cref="Problem"/> object using the labels provided.  If a label is set to "0" that item is ignored.
        /// </summary>
        /// <param name="rowLabels">The labels for the row items</param>
        /// <param name="columnLabels">The labels for the column items</param>
        /// <returns>A <see cref="Problem"/> object</returns>
        public Problem Compute(double[] rowLabels, double[] columnLabels)
        {
            List<Node[]> X = new List<Node[]>();
            List<double> Y = new List<double>();
            int maxIndex = 0;
            for (int i = 0; i < columnLabels.Length; i++)
            {
                if (columnLabels[i] != 0)
                {
                    maxIndex++;
                }
            }

            maxIndex++;
            for (int r = 0; r < rows; r++)
            {
                if (rowLabels[r] == 0)
                {
                    continue;
                }

                List<Node> nodes = new List<Node>();
                nodes.Add(new Node(0, X.Count + 1));
                for (int c = 0; c < columns; c++)
                {
                    if (columnLabels[c] == 0)
                    {
                        continue;
                    }

                    double value = similarities[r, c];
                    nodes.Add(new Node(nodes.Count, value));
                }

                X.Add(nodes.ToArray());
                Y.Add(rowLabels[r]);
            }

            return new Problem(X.Count, Y.ToArray(), X.ToArray(), maxIndex);
        }
    }
}
