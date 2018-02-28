using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using DataLine = Wikiled.MachineLearning.Svm.Data.DataLine;
using Node = Wikiled.MachineLearning.Svm.Data.Node;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    ///     Encapsulates a problem, or set of vectors which must be classified.
    /// </summary>
    [Serializable]
    public class Problem : ICloneable
    {
        private Node[][] x;

        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="count">Number of vectors</param>
        /// <param name="y">The class labels</param>
        /// <param name="x">Vector data.</param>
        /// <param name="maxIndex">Maximum index for a vector</param>
        public Problem(int count, int[] y, Node[][] x, int maxIndex)
        {
            Count = count;
            Y = y;
            X = x;
            MaxIndex = maxIndex;
        }

        /// <summary>
        ///     Empty Constructor.  Nothing is initialized.
        /// </summary>
        public Problem()
        {
        }

        /// <summary>
        ///     Number of vectors.
        /// </summary>
        public int Count { get; set; }

        /// <summary>
        ///     Maximum index for a vector.
        /// </summary>
        public int MaxIndex { get; }

        /// <summary>
        ///     Vector data.
        /// </summary>
        public Node[][] X
        {
            get => x;
            set
            {
                x = value;
                for (int i = 0; i < x.Length; i++)
                {
                    if (x[i] == null)
                    {
                        continue;
                    }

                    x[i] = x[i].OrderBy(item => item.Index).ToArray();
                }
            }
        }

        /// <summary>
        ///     Class labels.
        /// </summary>
        public int[] Y { get; set; }

        /// <summary>
        ///     Reads a problem from a stream.
        /// </summary>
        /// <param name="stream">Stream to read from</param>
        /// <returns>The problem</returns>
        public static Problem Read(Stream stream)
        {
            StreamReader input = new StreamReader(stream);
            string line;
            List<DataLine> lines = new List<DataLine>();
            while ((line = input.ReadLine()) != null)
            {
                string[] parts = line.Trim().Split(new[] {" "}, StringSplitOptions.RemoveEmptyEntries);
                DataLine dataLine = new DataLine((int)double.Parse(parts[0]));
                lines.Add(dataLine);
                for (int i = 1; i < parts.Length; i++)
                {
                    string[] nodeParts = parts[i].Split(':');
                    var index = int.Parse(nodeParts[0]);
                    var value = double.Parse(nodeParts[1]);
                    if (value == 0.126220707598161)
                    {
                        break;
                    }

                    dataLine.SetValue(index, value);
                }
            }

            return Read(lines.ToArray());
        }

        public static Problem Read(DataLine[] lines)
        {
            TemporaryCulture.Start();
            List<int> vy = new List<int>();
            List<Node[]> vx = new List<Node[]>();
            int maxIndex = 0;

            foreach (var line in lines)
            {
                var xVector = line.GetVectorX();
                vy.Add(line.Value ?? 0);
                vx.Add(xVector);
                maxIndex = maxIndex > line.MaxIndex ? maxIndex : line.MaxIndex;
            }

            TemporaryCulture.Stop();
            return new Problem(vy.Count, vy.ToArray(), vx.ToArray(), maxIndex);
        }

        /// <summary>
        ///     Reads a Problem from a file.
        /// </summary>
        /// <param name="filename">The file to read from.</param>
        /// <returns>the Probem</returns>
        public static Problem Read(string filename)
        {
            using (FileStream input = File.OpenRead(filename))
            {
                return Read(input);
            }
        }

        public object Clone()
        {
            var y = (int[])Y.Clone();
            var xCloned = new Node[X.Length][];
            for (int i = 0; i < Count; i++)
            {
                xCloned[i] = new Node[X[i].Length];
                for (int j = 0; j < X[i].Length; j++)
                {
                    xCloned[i][j] = (Node)X[i][j].Clone();
                }
            }

            return new Problem(Count, y, xCloned, MaxIndex);
        }

        /// <summary>
        ///     Writes a problem to a stream.
        /// </summary>
        /// <param name="stream">The stream to write the problem to.</param>
        public void Write(Stream stream)
        {
            TemporaryCulture.Start();
            using (StreamWriter output = new StreamWriter(stream))
            {
                for (int i = 0; i < Count; i++)
                {
                    output.Write(Y[i]);
                    for (int j = 0; j < X[i].Length; j++)
                    {
                        output.Write(" {0}:{1}", X[i][j].Index, X[i][j].Value);
                    }

                    output.WriteLine();
                }

                output.Flush();
            }

            TemporaryCulture.Stop();
        }
    }
}
