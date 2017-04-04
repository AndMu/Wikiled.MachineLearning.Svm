using System;
using NLog;
using Wikiled.Arff.Data;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    /// Class containing the routines to perform class membership prediction using a trained SVM.
    /// </summary>
    public static class Prediction
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        /// <summary>
        /// Predicts the class memberships of all the vectors in the problem.
        /// </summary>
        /// <param name="problem">The SVM Problem to solve</param>
        /// <param name="model">The Model to use</param>
        /// <param name="predictProbability">Whether to output a distribution over the classes</param>
        /// <returns>Percentage correctly labelled</returns>
        public static PredictionResult Predict(
            Problem problem,
            Model model,
            bool predictProbability)
        {
            int correct = 0;
            PredictionResult result = new PredictionResult();
            SvmType svmType = Procedures.SvmGetSvmType(model);
            int nrClass = Procedures.SvmGetNrClass(model);
            int[] labels = new int[nrClass];
            double[] probEstimates = null;

            if (predictProbability)
            {
                if (svmType == SvmType.EPSILON_SVR || svmType == SvmType.NU_SVR)
                {
                    log.Info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + Procedures.SvmGetSvrProbability(model));
                }
                else
                {
                    Procedures.SvmGetLabels(model, labels);
                    probEstimates = new double[nrClass];
                    for (int j = 0; j < nrClass; j++)
                    {
                        result.AddLabel(labels[j]);
                    }
                }
            }

            for (int i = 0; i < problem.Count; i++)
            {
                ClassificationClass item = new ClassificationClass();
                result.Set(item);

                item.Target = problem.Y[i];
                Node[] xValues = problem.X[i];

                if (predictProbability &&
                    (svmType == SvmType.C_SVC || svmType == SvmType.NU_SVC))
                {
                    item.Actual = Procedures.SvmPredictProbability(model, xValues, probEstimates);
                    for (int j = 0; j < nrClass; j++)
                    {
                        item.Add(probEstimates[j]);
                    }
                }
                else
                {
                    item.Actual = Procedures.SvmPredict(model, xValues);
                }

                if (item.Actual == item.Target)
                {
                    correct++;
                }
            }
            
            result.CorrectProbability = (double)correct / problem.Count;
            return result;
        }

        /// <summary>
        /// Predict the class for a single input vector.
        /// </summary>
        /// <param name="model">The Model to use for prediction</param>
        /// <param name="x">The vector for which to predict class</param>
        /// <returns>The result</returns>
        public static double Predict(Model model, Node[] x)
        {
            return Procedures.SvmPredict(model, x);
        }

        /// <summary>
        /// Predicts a class distribution for the single input vector.
        /// </summary>
        /// <param name="model">Model to use for prediction</param>
        /// <param name="x">The vector for which to predict the class distribution</param>
        /// <returns>A probability distribtion over classes</returns>
        public static double[] PredictProbability(Model model, Node[] x)
        {
            SvmType svmType = Procedures.SvmGetSvmType(model);
            if (svmType != SvmType.C_SVC && svmType != SvmType.NU_SVC)
            {
                throw new Exception("Model type " + svmType + " unable to predict probabilities.");
            }

            int nr_class = Procedures.SvmGetNrClass(model);
            double[] probEstimates = new double[nr_class];
            Procedures.SvmPredictProbability(model, x, probEstimates);
            return probEstimates;
        }
    }
}