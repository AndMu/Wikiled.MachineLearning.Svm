using Wikiled.Arff.Persistence;
using Wikiled.Common.Arguments;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public class TrainingResults
    {
        public TrainingResults(Model model, TrainingHeader header, IArffDataSet dataSet)
        {
            Guard.NotNull(() => model, model);
            Guard.NotNull(() => header, header);
            Guard.NotNull(() => dataSet, dataSet);
            Model = model;
            Header = header;
            DataSet = dataSet;
        }

        public IArffDataSet DataSet { get; }

        public Model Model { get; }

        public TrainingHeader Header { get; }
    }
}
