namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class ChangePoint
    {
        public ChangePoint(int tp, int fp, int tn, int fn)
        {
            TP = tp;
            FP = fp;
            TN = tn;
            FN = fn;
        }

        public int FN { get; }

        public int FP { get; }

        public int TN { get; }

        public int TP { get; }

        public override string ToString()
        {
            return $"{TP}:{FP}:{TN}:{FN}";
        }
    }
}