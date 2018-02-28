using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NLog;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    /// <summary>
    ///     This class contains routines which perform parameter selection for a model which uses C-SVC and
    ///     an RBF kernel.
    /// </summary>
    public class GridParameterSelection : IParameterSelection
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        private readonly TaskFactory taskFactory;

        private double crossValidation = double.MinValue;

        public GridParameterSelection(TaskFactory taskFactory, ITrainingModel training, GridSearchParameters parameters)
        {
            Guard.NotNull(() => parameters, parameters);
            Guard.NotNull(() => taskFactory, taskFactory);
            Guard.NotNull(() => training, training);
            SearchParameters = parameters;
            Training = training;
            this.taskFactory = taskFactory;
        }

        public GridSearchParameters SearchParameters { get; }

        public ITrainingModel Training { get; }

        public async Task<Parameter> Find(Problem problem, CancellationToken token)
        {
            log.Info("Starting Grid selection {0}...", SearchParameters);
            Guard.NotNull(() => problem, problem);
            crossValidation = double.MinValue;
            var parameter = (Parameter)SearchParameters.Default.Clone();
            List<Task<(Parameter Parameter, double Accuracy)>> tasks = new List<Task<(Parameter, double)>>();
            foreach (var gamma in SearchParameters.Gamma)
            {
                foreach (var value in SearchParameters.C)
                {
                    var gammaValue = gamma;
                    tasks.Add(taskFactory.StartNew(() => Search(gammaValue, value, problem, token), token));
                }
            }

            var results = await Task.WhenAll(tasks).ConfigureAwait(false);
            if (results.All(item => item.Parameter == null))
            {
                log.Warn("No results found");
                return parameter;
            }

            var bestResult = results.Where(item => item.Parameter != null).OrderByDescending(item => item.Accuracy).First();

            log.Info("Found best: C:{0} Gamma:{1} Result:{2:F2}", bestResult.Item1.C, bestResult.Item1.Gamma, bestResult.Item2);
            parameter.C = bestResult.Parameter.C;
            parameter.Gamma = bestResult.Parameter.Gamma;
            parameter.Performance = bestResult.Accuracy;
            return parameter;
        }

        private (Parameter parameter, double accuracy) Search(double gamma, double cValue, Problem problem, CancellationToken token)
        {
            if (token.IsCancellationRequested)
            {
                return (null, 0);
            }

            var localParameters = (Parameter)SearchParameters.Default.Clone();
            localParameters.Token = token;
            localParameters.C = cValue;
            localParameters.Gamma = gamma;
            var test = Training.PerformCrossValidation((Problem)problem.Clone(), localParameters, SearchParameters.Folds);
            if (test > crossValidation)
            {
                // possible race condition but we don't care it is just for logging - we don't use this value
                Interlocked.Exchange(ref crossValidation, test);
                log.Info("New MAXIMUM! C:{0} Gamma:{1} {2:F2}%", localParameters.C, localParameters.Gamma, test * 100);
            }
            else
            {
                log.Info("C:{0} Gamma:{1} {2:F2}%", localParameters.C, localParameters.Gamma, test * 100);
            }

            return (localParameters, test);
        }
    }
}
