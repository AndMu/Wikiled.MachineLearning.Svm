using Wikiled.Arff.Normalization;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class TrainingHeader
    {
        public double AverageVectorSize { get; set; }

        public bool GridSelection { get; set; }

        public KernelType Kernel { get; set; }

        public NormalizationType Normalization { get; set; }

        public SvmType SvmType { get; set; }

        public static TrainingHeader Create(KernelType type, SvmType svmType)
        {
            TrainingHeader header = new TrainingHeader();
            header.GridSelection = true;
            header.Normalization = NormalizationType.None;
            header.Kernel = type;
            header.SvmType = svmType;
            return header;
        }

        public static TrainingHeader CreateDefault()
        {
            return Create(KernelType.Linear, SvmType.C_SVC);
        }
    }
}
