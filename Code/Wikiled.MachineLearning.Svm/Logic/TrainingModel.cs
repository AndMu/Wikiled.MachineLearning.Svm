using System;
using NLog;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    ///     Class containing the routines to train SVM models.
    /// </summary>
    public class TrainingModel : ITrainingModel
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        /// <summary>
        ///     Performs cross validation.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to test</param>
        /// <param name="nrfold">The number of cross validations to use</param>
        /// <returns>The cross validation score</returns>
        public double PerformCrossValidation(Problem problem, Parameter parameters, int nrfold)
        {
            string error = Procedures.SvmCheckParameter(problem, parameters);
            if (error != null)
            {
                throw new Exception(error);
            }

            return DoCrossValidation(problem, parameters, nrfold);
        }

        /// <summary>
        ///     Trains a model using the provided training data and parameters.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to use</param>
        /// <returns>A trained SVM Model</returns>
        public Model Train(Problem problem, Parameter parameters = null)
        {
            if (parameters == null)
            {
                parameters = new Parameter();
            }

            log.Info("Training with {0}...", parameters);
            string error = Procedures.SvmCheckParameter(problem, parameters);
            if (error != null)
            {
                throw new Exception(error);
            }

            return Procedures.SvmTrain(problem, parameters);
        }

        private double DoCrossValidation(Problem problem, Parameter parameters, int nrFold)
        {
            int i;
            double[] target = new double[problem.Count];
            Procedures.SvmCrossValidation(problem, parameters, nrFold, target);
            int totalCorrect = 0;
            double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
            if (parameters.SvmType == SvmType.EPSILON_SVR || parameters.SvmType == SvmType.NU_SVR)
            {
                for (i = 0; i < problem.Count; i++)
                {
                    double y = problem.Y[i];
                    double v = target[i];
                    sumv += v;
                    sumy += y;
                    sumvv += v * v;
                    sumyy += y * y;
                    sumvy += v * y;
                }

                return (problem.Count * sumvy - sumv * sumy) / (Math.Sqrt(problem.Count * sumvv - sumv * sumv) * Math.Sqrt(problem.Count * sumyy - sumy * sumy));
            }

            for (i = 0; i < problem.Count; i++)
            {
                if (target[i] == problem.Y[i])
                {
                    ++totalCorrect;
                }
            }

            return (double)totalCorrect / problem.Count;
        }
    }
}
