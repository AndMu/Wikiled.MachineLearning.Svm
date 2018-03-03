using System;
using System.Globalization;
using System.IO;
using Node = Wikiled.MachineLearning.Svm.Data.Node;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    /// A transform which learns the mean and variance of a sample set and uses these to transform new data
    /// so that it has zero mean and unit variance.
    /// </summary>
    public class GaussianTransform : IRangeTransform
    {
        private readonly double[] means;

        private readonly double[] stddevs;

        /// <summary>
        /// Determines the Gaussian transform for the provided problem.
        /// </summary>
        /// <param name="prob">The Problem to analyze</param>
        /// <returns>The Gaussian transform for the problem</returns>
        public static GaussianTransform Compute(Problem prob)
        {
            int[] counts = new int[prob.MaxIndex];
            double[] means = new double[prob.MaxIndex];
            foreach (Node[] sample in prob.X)
            {
                for (int i = 0; i < sample.Length; i++)
                {
                    means[sample[i].Index - 1] += sample[i].Value;
                    counts[sample[i].Index - 1]++;
                }
            }

            for (int i = 0; i < prob.MaxIndex; i++)
            {
                if (counts[i] == 0)
                {
                    counts[i] = 2;
                }

                means[i] /= counts[i];
            }

            double[] stddevs = new double[prob.MaxIndex];
            foreach (Node[] sample in prob.X)
            {
                for (int i = 0; i < sample.Length; i++)
                {
                    double diff = sample[i].Value - means[sample[i].Index - 1];
                    stddevs[sample[i].Index - 1] += diff * diff;
                }
            }

            for (int i = 0; i < prob.MaxIndex; i++)
            {
                if (stddevs[i] == 0)
                {
                    continue;
                }
                stddevs[i] /= (counts[i] - 1);
                stddevs[i] = Math.Sqrt(stddevs[i]);
            }

            return new GaussianTransform(means, stddevs);
        }

        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="means">Means in each dimension</param>
        /// <param name="stddevs">Standard deviation in each dimension</param>
        public GaussianTransform(double[] means, double[] stddevs)
        {
            this.means = means;
            this.stddevs = stddevs;
        }

        /// <summary>
        /// Saves the transform to the disk.  The samples are not stored, only the 
        /// statistics.
        /// </summary>
        /// <param name="stream">The destination stream</param>
        public void Write(Stream stream)
        {
            TemporaryCulture.Start();

            using (StreamWriter output = new StreamWriter(stream))
            {
                output.WriteLine(means.Length);
                for (int i = 0; i < means.Length; i++)
                {
                    output.WriteLine("{0} {1}", means[i], stddevs[i]);
                }
                output.Flush();
            }
            TemporaryCulture.Stop();
        }

        /// <summary>
        /// Reads a GaussianTransform from the provided stream.
        /// </summary>
        /// <param name="stream">The source stream</param>
        /// <returns>The transform</returns>
        public static GaussianTransform Read(Stream stream)
        {
            TemporaryCulture.Start();

            using (StreamReader input = new StreamReader(stream))
            {
                int length = int.Parse(input.ReadLine(), CultureInfo.InvariantCulture);
                double[] means = new double[length];
                double[] stddevs = new double[length];
                for (int i = 0; i < length; i++)
                {
                    string[] parts = input.ReadLine().Split(new [] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                    means[i] = double.Parse(parts[0], CultureInfo.InvariantCulture);
                    stddevs[i] = double.Parse(parts[1], CultureInfo.InvariantCulture);
                }
                TemporaryCulture.Stop();

                return new GaussianTransform(means, stddevs);
            }
        }

        /// <summary>
        /// Saves the transform to the disk.  The samples are not stored, only the 
        /// statistics.
        /// </summary>
        /// <param name="filename">The destination filename</param>
        /// <param name="transform">The transform</param>
        public void Write(string filename, GaussianTransform transform)
        {
            using (FileStream output = File.Open(filename, FileMode.Create))
            {
                Write(output);
                output.Close();
            }
        }

        /// <summary>
        /// Reads a GaussianTransform from the provided stream.
        /// </summary>
        /// <param name="filename">The source filename</param>
        /// <returns>The transform</returns>
        public static GaussianTransform Read(string filename)
        {
            using (FileStream input = File.Open(filename, FileMode.Open))
            {
                return Read(input);
            }
        }

        /// <summary>
        /// Transform the input value using the transform stored for the provided index.
        /// </summary>
        /// <param name="input">Input value</param>
        /// <param name="index">Index of the transform to use</param>
        /// <returns>The transformed value</returns>
        public double Transform(double input, int index)
        {
            index--;
            if (stddevs[index] == 0)
            {
                return 0;
            }
            double diff = input - means[index];
            diff /= stddevs[index];
            return diff;
        }
        /// <summary>
        /// Transforms the input array.
        /// </summary>
        /// <param name="input">The array to transform</param>
        /// <returns>The transformed array</returns>
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
    }
}
