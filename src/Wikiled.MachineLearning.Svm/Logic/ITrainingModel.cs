using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    public interface ITrainingModel
    {
        TrainingHeader Header { get; }

        /// <summary>
        ///     Performs cross validation.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to test</param>
        /// <param name="nrfold">The number of cross validations to use</param>
        /// <returns>The cross validation score</returns>
        double PerformCrossValidation(Problem problem, Parameter parameters, int nrfold);

        /// <summary>
        ///     Trains a model using the provided training data and parameters.
        /// </summary>
        /// <param name="problem">The training data</param>
        /// <param name="parameters">The parameters to use</param>
        /// <returns>A trained SVM Model</returns>
        Model Train(Problem problem, Parameter parameters = null);
    }
}