namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class SolverParameters
    {
        public SolverParameters(
            int totalProblems,
            IQMatrix qMatrix,
            double[] p,
            sbyte[] y,
            double[] alpha,
            double cp,
            double cn,
            double eps,
            SolutionInfo solutionInfo,
            bool shrinking)
        {
            TotalProblems = totalProblems;
            QMatrix = qMatrix;
            P = p;
            Y = y;
            Alpha = alpha;
            Cn = cn;
            Cp = cp;
            Eps = eps;
            SolutionInfo = solutionInfo;
            Shrinking = shrinking;
        }

        public double[] Alpha { get; }

        public double Cn { get; }

        public double Cp { get; }

        public double Eps { get; }

        public int? MaxIterations { get; set; }

        public double[] P { get; }

        public IQMatrix QMatrix { get; }

        public bool Shrinking { get; }

        public SolutionInfo SolutionInfo { get; }

        public int TotalProblems { get; }

        public sbyte[] Y { get; }
    }
}
