using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using NLog;
using Wikiled.Arff.Persistence;
using Wikiled.Core.Utility.Arguments;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    public class ParametersSelectionFactory
    {
        private readonly TaskFactory taskFactory;

        private readonly Logger logger = LogManager.GetCurrentClassLogger();

        public ParametersSelectionFactory(TaskFactory taskFactory)
        {
            Guard.NotNull(() => taskFactory, taskFactory);
            this.taskFactory = taskFactory;
        }

        public IParameterSelection Create(TrainingHeader header, IArffDataSet dataset)
        {
            Guard.NotNull(() => header, header);
            Guard.NotNull(() => dataset, dataset);
            Parameter defaultParameter = new Parameter();
            defaultParameter.KernelType = header.Kernel;
            defaultParameter.CacheSize = 200;
            defaultParameter.SvmType = header.SvmType;
            if (!header.GridSelection)
            {
                return new NullParameterSelection(defaultParameter);
            }

            GridSearchParameters searchParameters;
            logger.Info("Investigate LibLinear");
            if (header.Kernel == KernelType.Linear)
            {
                var gamma = GetList(1, 1, 1);
                if (dataset.Header.Total > (dataset.TotalDocuments * 10))
                {
                    logger.Info("Selecting Linear features >> instances");
                    defaultParameter.Shrinking = false;

                }
                else
                {
                    logger.Warn("Investigate LibLinear");
                }

                var training = dataset.GetProblem();
                var weights = WeightCalculation.GetWeights(training.Y);
                foreach (var classItem in weights)
                {
                    logger.Info($"Using class [{classItem.Key}] with weight [{classItem.Value}]");
                }

                searchParameters = new GridSearchParameters(5, GetList(-1, 2, 1), gamma, defaultParameter);
            }
            else
            {
                searchParameters = new GridSearchParameters(5, GetList(-5, 15, 2), GetList(-15, 3, 2), defaultParameter);
            }

            return new GridParameterSelection(taskFactory, new TrainingModel(), searchParameters);
        }

        private double[] GetList(double minPower, double maxPower, double iteration)
        {
            List<double> list = new List<double>();
            for (double d = minPower; d <= maxPower; d += iteration)
            {
                list.Add(Math.Pow(2, d));
            }

            return list.ToArray();
        }
    }
}