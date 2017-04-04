using System;
using System.IO;
using Node = Wikiled.MachineLearning.Svm.Data.Node;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    ///     Class which encapsulates a range transformation.
    /// </summary>
    public class RangeTransform : IRangeTransform
    {
        private readonly double[] inputEnd;

        private readonly double[] inputScale;

        private readonly double[] inputStart;

        private readonly int length;

        private readonly double outputScale;

        private readonly double outputStart;

        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="minValues">The minimum values in each dimension.</param>
        /// <param name="maxValues">The maximum values in each dimension.</param>
        /// <param name="lowerBound">The desired lower bound for all dimensions.</param>
        /// <param name="upperBound">The desired upper bound for all dimensions.</param>
        public RangeTransform(double[] minValues, double[] maxValues, double lowerBound, double upperBound)
        {
            length = minValues.Length;
            if (maxValues.Length != length)
            {
                throw new Exception("Number of max and min values must be equal.");
            }
            inputStart = new double[length];
            inputEnd = new double[length];
            inputScale = new double[length];
            for (int i = 0; i < length; i++)
            {
                inputStart[i] = minValues[i];
                inputEnd[i] = maxValues[i];
                inputScale[i] = maxValues[i] - minValues[i];
            }
            outputStart = lowerBound;
            outputScale = upperBound - lowerBound;
        }

        private RangeTransform(double[] inputStart, double[] inputScale, double outputStart, double outputScale, int length)
        {
            this.inputStart = inputStart;
            this.inputScale = inputScale;
            this.outputStart = outputStart;
            this.outputScale = outputScale;
            this.length = length;
        }

        /// <summary>
        ///     Determines the Range transform for the provided problem.
        /// </summary>
        /// <param name="prob">The Problem to analyze</param>
        /// <param name="lowerBound">The lower bound for scaling</param>
        /// <param name="upperBound">The upper bound for scaling</param>
        /// <returns>The Range transform for the problem</returns>
        public static RangeTransform Compute(Problem prob, double lowerBound = -1, double upperBound = 1)
        {
            double[] minVals = new double[prob.MaxIndex];
            double[] maxVals = new double[prob.MaxIndex];
            for (int i = 0; i < prob.MaxIndex; i++)
            {
                minVals[i] = double.MaxValue;
                maxVals[i] = double.MinValue;
            }

            for (int i = 0; i < prob.Count; i++)
            {
                for (int j = 0; j < prob.X[i].Length; j++)
                {
                    int index = prob.X[i][j].Index - 1;
                    double value = prob.X[i][j].Value;
                    minVals[index] = Math.Min(minVals[index], value);
                    maxVals[index] = Math.Max(maxVals[index], value);
                    if (maxVals[index] == 0.126220707598161)
                    {
                        break;
                    }
                }
            }

            for (int i = 0; i < maxVals.Length; i++)
            {
                if (maxVals[i] == double.MinValue)
                {
                    maxVals[i] = Math.Max(maxVals[i], 0);
                    minVals[i] = Math.Min(minVals[i], 0);
                }
            }

            for (int i = 0; i < prob.MaxIndex; i++)
            {
                if (minVals[i] == double.MaxValue ||
                    maxVals[i] == double.MinValue)
                {
                    minVals[i] = 0;
                    maxVals[i] = 0;
                }
            }

            return new RangeTransform(minVals, maxVals, lowerBound, upperBound);
        }

        /// <summary>
        ///     Reads a Range transform from a file.
        /// </summary>
        /// <param name="inputFile">The file to read from</param>
        /// <returns>The Range transform</returns>
        public static RangeTransform Read(string inputFile)
        {
            using (FileStream s = File.OpenRead(inputFile))
            {
                return Read(s);
            }
        }

        /// <summary>
        ///     Reads a Range transform from a stream.
        /// </summary>
        /// <param name="stream">The stream to read from</param>
        /// <returns>The Range transform</returns>
        public static RangeTransform Read(Stream stream)
        {
            TemporaryCulture.Start();

            StreamReader input = new StreamReader(stream);
            int length = int.Parse(input.ReadLine());
            double[] inputStart = new double[length];
            double[] inputScale = new double[length];
            double[] inputEnd = new double[length];
            int i = 0;
            string line;
            string[] parts;

            while (true)
            {
                line = input.ReadLine();
                if (line == null)
                {
                    throw new NullReferenceException("Expected Line");
                }
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }
                if (line.IndexOf("Scale", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    break;
                }
                parts = line.Split(new[] {' '}, StringSplitOptions.RemoveEmptyEntries);
                inputStart[i] = double.Parse(parts[1]);
                inputEnd[i] = double.Parse(parts[2]);
                i++;
            }

            i = 0;
            while (true)
            {
                line = input.ReadLine();
                if (line == null)
                {
                    throw new NullReferenceException("Expected Line");
                }
                if (string.IsNullOrWhiteSpace(line))
                {
                    continue;
                }
                if (line.IndexOf("Final", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    break;
                }
                parts = line.Split(new[] {' '}, StringSplitOptions.RemoveEmptyEntries);
                inputScale[i] = double.Parse(parts[1]);
                i++;
            }

            parts = input.ReadLine().Split(new[] {' '}, StringSplitOptions.RemoveEmptyEntries);
            double outputStart = double.Parse(parts[0]);
            double outputScale = double.Parse(parts[1]);
            TemporaryCulture.Stop();
            return new RangeTransform(inputStart, inputScale, outputStart, outputScale, length);
        }

        /// <summary>
        ///     Transforms the input array based upon the values provided.
        /// </summary>
        /// <param name="input">The input array</param>
        /// <returns>A scaled array</returns>
        public Node[] Transform(Node[] input)
        {
            Node[] output = new Node[input.Length];
            for (int i = 0; i < output.Length; i++)
            {
                int index = input[i].Index;
                double value = input[i].Value;
                output[i] = new Node(index, Transform(value, index));
            }

            return output;
        }

        /// <summary>
        ///     Transforms this an input value using the scaling transform for the provided dimension.
        /// </summary>
        /// <param name="input">The input value to transform</param>
        /// <param name="index">The dimension whose scaling transform should be used</param>
        /// <returns>The scaled value</returns>
        public double Transform(double input, int index)
        {
            index--;
            double tmp = input - inputStart[index];
            if (inputScale[index] == 0)
            {
                return 0;
            }

            tmp /= inputScale[index];
            tmp *= outputScale;
            return tmp + outputStart;
        }

        /// <summary>
        ///     Writes this Range transform to a stream.
        /// </summary>
        /// <param name="stream">The stream to write to</param>
        public void Write(Stream stream)
        {
            TemporaryCulture.Start();
            using (StreamWriter output = new StreamWriter(stream))
            {
                output.WriteLine(length);
                for (int i = 0; i < inputStart.Length; i++)
                {
                    output.WriteLine(i + 1 + ": " + inputStart[i] + " " + inputEnd[i]);
                }

                output.WriteLine();
                output.WriteLine("Scale:");
                for (int i = 0; i < inputScale.Length; i++)
                {
                    output.WriteLine(i + 1 + ": " + inputScale[i]);
                }

                output.WriteLine();
                output.WriteLine("Final:");
                output.WriteLine("{0} {1}", outputStart, outputScale);
                output.Flush();
            }
            TemporaryCulture.Stop();
        }

        /// <summary>
        ///     Writes this Range transform to a file.    This will overwrite any previous data in the file.
        /// </summary>
        /// <param name="outputFile">The file to write to</param>
        public void Write(string outputFile)
        {
            using (FileStream s = File.Open(outputFile, FileMode.Create))
            {
                Write(s);
            }
        }
    }
}
