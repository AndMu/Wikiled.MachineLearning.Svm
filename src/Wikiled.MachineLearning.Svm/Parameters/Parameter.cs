using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    /// <summary>
    ///     This class contains the various parameters which can affect the way in which an SVM
    ///     is learned.  Unless you know what you are doing, chances are you are best off using
    ///     the default values.
    /// </summary>
    [Serializable]
    public class Parameter : ICloneable
    {
        /// <summary>
        ///     DefaultParallel Constructor.  Gives good default values to all parameters.
        /// </summary>
        public Parameter()
        {
            SvmType = SvmType.C_SVC;
            KernelType = KernelType.Linear;
            Degree = 3;
            Gamma = 0; // 1/k
            Coefficient0 = 0;
            Nu = 0.5;
            CacheSize = 40;
            C = 1;
            EPS = 1e-3;
            P = 0.1;
            Shrinking = true;
            Probability = false;
            Weights = new Dictionary<int, double>();
        }

        /// <summary>
        ///     The parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        /// </summary>
        public double C { get; set; }

        /// <summary>
        ///     Cache memory size in MB (default 100)
        /// </summary>
        public double CacheSize { get; set; }

        /// <summary>
        ///     Zeroeth coefficient in kernel function (default 0)
        /// </summary>
        public double Coefficient0 { get; set; }

        /// <summary>
        ///     Degree in kernel function (default 3).
        /// </summary>
        public int Degree { get; set; }

        /// <summary>
        ///     Tolerance of termination criterion (default 0.001)
        /// </summary>
        public double EPS { get; set; }

        /// <summary>
        ///     Gamma in kernel function (default 1/k)
        /// </summary>
        public double Gamma { get; set; }

        /// <summary>
        ///     Type of kernel function (default Polynomial)
        /// </summary>
        public KernelType KernelType { get; set; }

        /// <summary>
        ///     The parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
        /// </summary>
        public double Nu { get; set; }

        /// <summary>
        ///     The epsilon in loss function of epsilon-SVR (default 0.1)
        /// </summary>
        public double P { get; set; }

        /// <summary>
        ///     Parameter performance
        /// </summary>
        public double Performance { get; set; }

        /// <summary>
        ///     Whether to train an SVC or SVR model for probability estimates, (default False)
        /// </summary>
        public bool Probability { get; set; }

        /// <summary>
        ///     Whether to use the shrinking heuristics, (default True)
        /// </summary>
        public bool Shrinking { get; set; }

        /// <summary>
        ///     Type of SVM (default C-SVC)
        /// </summary>
        public SvmType SvmType { get; set; }

        public CancellationToken Token { get; set; }

        /// <summary>
        ///     Contains custom weights for class labels.  DefaultParallel weight value is 1.
        /// </summary>
        public Dictionary<int, double> Weights { get; set; }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            builder.AppendFormat(
                "Parameter: Type:{0} Kernel:{1} Degree:{2} Gamma:{3} Coof:{4} Nu:{5} Cache:{6}, C:{7} EPS:{8} P:{9} Shrinking:{10} Probability:{11}",
                SvmType,
                KernelType,
                Degree,
                Gamma,
                Coefficient0,
                Nu,
                CacheSize,
                C,
                EPS,
                P,
                Shrinking,
                Probability);

            foreach (var weight in Weights)
            {
                builder.AppendFormat(" Weight:{0}={1}", weight.Key, weight.Value);
            }

            return builder.ToString();
        }

        /// <summary>
        ///     Creates a memberwise clone of this parameters object.
        /// </summary>
        /// <returns>The clone (as type Parameter)</returns>
        public object Clone()
        {
            return MemberwiseClone();
        }
    }
}
