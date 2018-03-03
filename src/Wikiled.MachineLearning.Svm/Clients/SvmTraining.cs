using System;
using System.Threading;
using System.Threading.Tasks;
using NLog;
using Wikiled.Arff.Persistence;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Data;
using Wikiled.MachineLearning.Svm.Logic;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Clients
{
    public class SvmTraining : ISvmTraining
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private readonly IProblemFactory problemFactory;

        private readonly IArffDataSet dataSet;

        public SvmTraining(IProblemFactory problemFactory, IArffDataSet dataSet)
        {
            Guard.NotNull(() => problemFactory, problemFactory);
            Guard.NotNull(() => dataSet, dataSet);
            this.problemFactory = problemFactory;
            this.dataSet = dataSet;
            if (dataSet.RandomSeed == null)
            {
                dataSet.RandomSeed = Environment.TickCount;
            }
        }

        public IParameterSelection SelectParameters(TrainingHeader header, CancellationToken token)
        {
            log.Info("Selecting parameters...");
            if (dataSet.TotalDocuments == 0)
            {
                log.Error("No document found");
                return null;
            }

            var scheduler = new ConcurrentExclusiveSchedulerPair(TaskScheduler.Default, Environment.ProcessorCount / 2)
                .ConcurrentScheduler;
            var taskFactory = new TaskFactory(
                token,
                TaskCreationOptions.LongRunning,
                TaskContinuationOptions.LongRunning,
                scheduler);

            // https://www.quora.com/Support-Vector-Machines/SVM-performance-depends-on-scaling-and-normalization-Is-this-considered-a-drawback
            header.Normalization = dataSet.Normalization;

            ParametersSelectionFactory factory = new ParametersSelectionFactory(taskFactory, problemFactory);
            var selection = factory.Create(header, dataSet);
            return selection;
        }

        public async Task<TrainingResults> Train(IParameterSelection selection)
        {
            Guard.NotNull(() => selection, selection);
            Problem problem = problemFactory.Construct(dataSet).GetProblem();
            var parameters = await selection.Find(problem, CancellationToken.None).ConfigureAwait(false);

            // it is reasonable to choose values between 1 and 10^15
            // http://stackoverflow.com/questions/19089913/data-imbalance-in-svm-using-libsvm
            // http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
            // http://stats.stackexchange.com/questions/31066/what-is-the-influence-of-c-in-svms-with-linear-kernel
            log.Info("Training...");
            var result = selection.Training.Train(problem, parameters);
            log.Info("Training Done.");
            return new TrainingResults(result, selection.Training.Header, dataSet);
        }
    }
}
