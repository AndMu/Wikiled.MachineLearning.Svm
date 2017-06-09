using Wikiled.Arff.Normalization;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class TrainingHeader
    {
        public static TrainingHeader CreateDefault()
        {
            return Create(KernelType.Linear, SvmType.C_SVC);
        }

        public static TrainingHeader Create(KernelType type, SvmType svmType)
        {
            TrainingHeader header = new TrainingHeader();
            header.GridSelection = true;
            header.Normalization = NormalizationType.None;
            header.Kernel = type;
            header.SvmType = svmType;
            return header;
        }

        public bool GridSelection { get; set; }

        public double AverageVectorSize { get; set; }

        public NormalizationType Normalization { get; set; }

        public KernelType Kernel { get; set; }

        public SvmType SvmType { get; set; }
    }
}
