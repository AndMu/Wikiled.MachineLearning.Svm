using Wikiled.Common.Arguments;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    public class GridSearchParameters
    {
        public GridSearchParameters(
            int folds,
            double[] c,
            double[] gamma,
            Parameter parameter)
        {
            Guard.NotNull(() => parameter, parameter);
            Folds = folds;
            C = c;
            Gamma = gamma;
            Default = parameter;
        }

        public int Folds { get; }

        public double[] C { get; }

        public double[] Gamma { get; }

        public Parameter Default { get; }

        public override string ToString()
        {
            return $"Grid search paramaters. Folds:{Folds} C:{C.Length} Gamma:{Gamma.Length}";
        }
    }
}