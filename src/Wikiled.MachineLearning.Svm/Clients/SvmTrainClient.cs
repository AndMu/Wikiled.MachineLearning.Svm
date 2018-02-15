using System;
using System.Threading;
using System.Threading.Tasks;
using NLog;
using Wikiled.Arff.Persistence;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public class SvmTrainClient : ISvmTrain
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private readonly IArffDataSet dataSet;

        public SvmTrainClient(IArffDataSet dataSet)
        {
            Guard.NotNull(() => dataSet, dataSet);
            this.dataSet = dataSet;
        }

        public async Task<TrainingResults> Train(TrainingHeader header, CancellationToken token)
        {
            if (dataSet.TotalDocuments == 0)
            {
                return null;
            }

            log.Info("Selecting parameters...");
            // https://www.quora.com/Support-Vector-Machines/SVM-performance-depends-on-scaling-and-normalization-Is-this-considered-a-drawback
            header.Normalization = dataSet.Normalization;
            Problem problem = dataSet.GetProblem();
            var scheduler = new ConcurrentExclusiveSchedulerPair(TaskScheduler.Default, Environment.ProcessorCount / 2)
                .ConcurrentScheduler;
            var taskFactory = new TaskFactory(
                token,
                TaskCreationOptions.LongRunning,
                TaskContinuationOptions.LongRunning,
                scheduler);
            
            ParametersSelectionFactory factory = new ParametersSelectionFactory(taskFactory);
            var selection = factory.Create(header, dataSet);
            var parameters = await selection.Find(problem, CancellationToken.None).ConfigureAwait(false);

            ////it is reasonable to choose values between 1 and 10^15
            ////http://stackoverflow.com/questions/19089913/data-imbalance-in-svm-using-libsvm
            // http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
            // http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
            log.Info("Training...");
            TrainingModel training = new TrainingModel();
            var result = training.Train(problem, parameters);
            log.Info("Training Done.");
            return new TrainingResults(result, header, dataSet);
        }
    }
}
