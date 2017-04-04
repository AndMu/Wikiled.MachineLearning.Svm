using System;
using System.IO;
using Wikiled.Arff.Data;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    /// Encapsulates an SVM Model.
    /// </summary>
    [Serializable]
    public class Model
    {
        /// <summary>
        /// Parameter object.
        /// </summary>
        public Parameter Parameter { get; set; }

        /// <summary>
        /// Number of classes in the model.
        /// </summary>
        public int NumberOfClasses { get; set; }

        /// <summary>
        /// Total number of support vectors.
        /// </summary>
        public int SupportVectorCount { get; set; }

        /// <summary>
        /// The support vectors.
        /// </summary>
        public Node[][] SupportVectors { get; set; }

        /// <summary>
        /// The coefficients for the support vectors.
        /// </summary>
        public double[][] SupportVectorCoefficients { get; set; }

        /// <summary>
        /// Rho values.
        /// </summary>
        public double[] Rho { get; set; }

        /// <summary>
        /// First pairwise probability.
        /// </summary>
        public double[] PairwiseProbabilityA { get; set; }

        /// <summary>
        /// Second pairwise probability.
        /// </summary>
        public double[] PairwiseProbabilityB { get; set; }

        // for classification only

        /// <summary>
        /// Class labels.
        /// </summary>
        public int[] ClassLabels { get; set; }

        /// <summary>
        /// Number of support vectors per class.
        /// </summary>
        public int[] NumberOfSVPerClass { get; set; }

        /// <summary>
        /// Reads a Model from the provided file.
        /// </summary>
        /// <param name="filename">The name of the file containing the Model</param>
        /// <returns>the Model</returns>
        public static Model Read(string filename)
        {
            FileStream input = File.OpenRead(filename);
            try
            {
                return Read(input);
            }
            finally
            {
                input.Close();
            }
        }

        /// <summary>
        /// Reads a Model from the provided stream.
        /// </summary>
        /// <param name="stream">The stream from which to read the Model.</param>
        /// <returns>the Model</returns>
        public static Model Read(Stream stream)
        {
            TemporaryCulture.Start();
            using (StreamReader input = new StreamReader(stream))
            {
                // read parameters
                Model model = new Model();
                Parameter param = new Parameter();
                model.Parameter = param;
                model.Rho = null;
                model.PairwiseProbabilityA = null;
                model.PairwiseProbabilityB = null;
                model.ClassLabels = null;
                model.NumberOfSVPerClass = null;

                bool headerFinished = false;
                while (!headerFinished)
                {
                    string line = input.ReadLine();
                    string cmd, arg;
                    int splitIndex = line.IndexOf(' ');
                    if (splitIndex >= 0)
                    {
                        cmd = line.Substring(0, splitIndex);
                        arg = line.Substring(splitIndex + 1);
                    }
                    else
                    {
                        cmd = line;
                        arg = "";
                    }

                    arg = arg.ToLower();

                    int i, n;
                    switch (cmd)
                    {
                        case "svm_type":
                            param.SvmType = (SvmType)Enum.Parse(typeof(SvmType), arg, true);
                            break;

                        case "performance":
                            param.Performance = double.Parse(arg);
                            break;

                        case "kernel_type":
                            param.KernelType = (KernelType)Enum.Parse(typeof(KernelType), arg, true);
                            break;

                        case "degree":
                            param.Degree = int.Parse(arg);
                            break;

                        case "gamma":
                            param.Gamma = double.Parse(arg);
                            break;

                        case "coef0":
                            param.Coefficient0 = double.Parse(arg);
                            break;

                        case "nr_class":
                            model.NumberOfClasses = int.Parse(arg);
                            break;

                        case "total_sv":
                            model.SupportVectorCount = int.Parse(arg);
                            break;

                        case "rho":
                            n = model.NumberOfClasses * (model.NumberOfClasses - 1) / 2;
                            model.Rho = new double[n];
                            string[] rhoParts = arg.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            for (i = 0; i < n; i++)
                            {
                                model.Rho[i] = double.Parse(rhoParts[i]);
                            }

                            break;
                        case "label":
                            n = model.NumberOfClasses;
                            model.ClassLabels = new int[n];
                            string[] labelParts = arg.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            for (i = 0; i < n; i++)
                            {
                                model.ClassLabels[i] = int.Parse(labelParts[i]);
                            }
                            break;

                        case "probA":
                            n = model.NumberOfClasses * (model.NumberOfClasses - 1) / 2;
                            model.PairwiseProbabilityA = new double[n];
                            string[] probAParts = arg.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            for (i = 0; i < n; i++)
                            {
                                model.PairwiseProbabilityA[i] = double.Parse(probAParts[i]);
                            }
                            break;

                        case "probB":
                            n = model.NumberOfClasses * (model.NumberOfClasses - 1) / 2;
                            model.PairwiseProbabilityB = new double[n];
                            string[] probBParts = arg.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            for (i = 0; i < n; i++)
                            {
                                model.PairwiseProbabilityB[i] = double.Parse(probBParts[i]);
                            }
                            break;

                        case "nr_sv":
                            n = model.NumberOfClasses;
                            model.NumberOfSVPerClass = new int[n];
                            string[] nrsvParts = arg.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                            for (i = 0; i < n; i++)
                            {
                                model.NumberOfSVPerClass[i] = int.Parse(nrsvParts[i]);
                            }
                            break;

                        case "SV":
                            headerFinished = true;
                            break;

                        default:
                            throw new Exception("Unknown text in model file");
                    }
                }

                // read sv_coef and SV
                int m = model.NumberOfClasses - 1;
                int l = model.SupportVectorCount;
                model.SupportVectorCoefficients = new double[m][];
                for (int i = 0; i < m; i++)
                {
                    model.SupportVectorCoefficients[i] = new double[l];
                }

                model.SupportVectors = new Node[l][];
                for (int i = 0; i < l; i++)
                {
                    string[] parts = input.ReadLine().Trim().Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

                    for (int k = 0; k < m; k++)
                    {
                        model.SupportVectorCoefficients[k][i] = double.Parse(parts[k]);
                    }

                    int n = parts.Length - m;
                    model.SupportVectors[i] = new Node[n];
                    for (int j = 0; j < n; j++)
                    {
                        string[] nodeParts = parts[m + j].Split(':');
                        model.SupportVectors[i][j] = new Node
                        {
                            Index = int.Parse(nodeParts[0]),
                            Value = double.Parse(nodeParts[1])
                        };
                    }
                }

                TemporaryCulture.Stop();
                return model;
            }
        }

        /// <summary>
        /// Writes a model to the provided filename.  This will overwrite any previous data in the file.
        /// </summary>
        /// <param name="filename">The desired file</param>
        public void Write(string filename)
        {
            using (FileStream stream = File.Open(filename, FileMode.Create))
            {
                Write(stream);
            }
        }

        /// <summary>
        /// Writes a model to the provided stream.
        /// </summary>
        /// <param name="stream">The output stream</param>
        public void Write(Stream stream)
        {
            TemporaryCulture.Start();

            using (StreamWriter output = new StreamWriter(stream))
            {
                Parameter param = Parameter;
                output.WriteLine($"svm_type {param.SvmType }");
                output.WriteLine($"kernel_type {param.KernelType }");
                if (param.KernelType == KernelType.Polynomial)
                {
                    output.WriteLine($"degree {param.Degree}");
                }

                if (param.KernelType == KernelType.Polynomial || param.KernelType == KernelType.RBF ||
                    param.KernelType == KernelType.Sigmoid)
                {
                    output.WriteLine($"gamma {param.Gamma}");
                }

                if (param.KernelType == KernelType.Polynomial || param.KernelType == KernelType.Sigmoid)
                {
                    output.WriteLine($"coef0 {param.Coefficient0}");
                }

                output.WriteLine($"performance {param.Performance}");

                int nrClass = NumberOfClasses;
                int l = SupportVectorCount;
                output.WriteLine($"nr_class {nrClass}");
                output.WriteLine($"total_sv {l}");
                output.Write("rho");

                for (int i = 0; i < nrClass * (nrClass - 1) / 2; i++)
                {
                    output.Write(" " + Rho[i]);
                }

                output.WriteLine();

                if (ClassLabels != null)
                {
                    output.Write("label");
                    for (int i = 0; i < nrClass; i++)
                    {
                        output.Write(" " + ClassLabels[i]);
                    }

                    output.WriteLine();
                }

                if (PairwiseProbabilityA != null)
                // regression has probA only
                {
                    output.Write("probA");
                    for (int i = 0; i < nrClass * (nrClass - 1) / 2; i++)
                    {
                        output.Write(" " + PairwiseProbabilityA[i]);
                    }

                    output.WriteLine();
                }

                if (PairwiseProbabilityB != null)
                {
                    output.Write("probB");
                    for (int i = 0; i < nrClass * (nrClass - 1) / 2; i++)
                    {
                        output.Write(" " + PairwiseProbabilityB[i]);
                    }

                    output.WriteLine();
                }

                if (NumberOfSVPerClass != null)
                {
                    output.Write("nr_sv");
                    for (int i = 0; i < nrClass; i++)
                    {
                        output.Write(" " + NumberOfSVPerClass[i]);
                    }

                    output.WriteLine();
                }

                output.WriteLine("SV");
                double[][] svCoef = SupportVectorCoefficients;
                Node[][] supportVectors = SupportVectors;

                for (int i = 0; i < l; i++)
                {
                    for (int j = 0; j < nrClass - 1; j++)
                    {
                        output.Write(svCoef[j][i] + " ");
                    }

                    Node[] p = supportVectors[i];
                    if (p.Length == 0)
                    {
                        output.WriteLine();
                        continue;
                    }

                    if (param.KernelType == KernelType.Precomputed)
                    {
                        output.Write("0:{0}", (int)p[0].Value);
                    }
                    else
                    {
                        output.Write("{0}:{1}", p[0].Index, p[0].Value);
                        for (int j = 1; j < p.Length; j++)
                        {
                            output.Write(" {0}:{1}", p[j].Index, p[j].Value);
                        }
                    }

                    output.WriteLine();
                }

                output.Flush();
            }

            TemporaryCulture.Stop();
        }
    }
}