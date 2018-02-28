using System;
using System.Linq;
using System.Threading.Tasks;
using NLog;
using Wikiled.MachineLearning.Svm.Parameters;
using Node = Wikiled.MachineLearning.Svm.Data.Node;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class Procedures
    {
        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        public static string SvmCheckParameter(Problem prob, Parameter param)
        {
            // svm_type
            SvmType svmType = param.SvmType;

            // kernel_type, degree
            if (param.Degree < 0)
            {
                return "degree of polynomial kernel < 0";
            }

            // cache_size,eps,C,nu,p,shrinking
            if (param.CacheSize <= 0)
            {
                return "cache_size <= 0";
            }

            if (param.EPS <= 0)
            {
                return "eps <= 0";
            }

            if (param.Gamma == 0)
            {
                param.Gamma = 1.0 / prob.MaxIndex;
            }

            if (svmType == SvmType.C_SVC ||
                svmType == SvmType.EPSILON_SVR ||
                svmType == SvmType.NU_SVR)
            {
                if (param.C <= 0)
                {
                    return "C <= 0";
                }
            }

            if (svmType == SvmType.NU_SVC ||
                svmType == SvmType.ONE_CLASS ||
                svmType == SvmType.NU_SVR)
            {
                if (param.Nu <= 0 || param.Nu > 1)
                {
                    return "nu <= 0 or nu > 1";
                }
            }

            if (svmType == SvmType.EPSILON_SVR)
            {
                if (param.P < 0)
                {
                    return "p < 0";
                }
            }

            if (param.Probability &&
                svmType == SvmType.ONE_CLASS)
            {
                return "one-class SVM probability output not supported yet";
            }

            // check whether nu-svc is feasible
            if (svmType == SvmType.NU_SVC)
            {
                int l = prob.Count;
                int maxNrClass = 16;
                int nrClass = 0;
                int[] label = new int[maxNrClass];
                int[] count = new int[maxNrClass];

                int i;
                for (i = 0; i < l; i++)
                {
                    int thisLabel = (int)prob.Y[i];
                    int j;
                    for (j = 0; j < nrClass; j++)
                    {
                        if (thisLabel == label[j])
                        {
                            ++count[j];
                            break;
                        }
                    }

                    if (j == nrClass)
                    {
                        if (nrClass == maxNrClass)
                        {
                            maxNrClass *= 2;
                            int[] newData = new int[maxNrClass];
                            Array.Copy(label, 0, newData, 0, label.Length);
                            label = newData;

                            newData = new int[maxNrClass];
                            Array.Copy(count, 0, newData, 0, count.Length);
                            count = newData;
                        }

                        label[nrClass] = thisLabel;
                        count[nrClass] = 1;
                        ++nrClass;
                    }
                }

                for (i = 0; i < nrClass; i++)
                {
                    int n1 = count[i];
                    for (int j = i + 1; j < nrClass; j++)
                    {
                        int n2 = count[j];
                        if (param.Nu * (n1 + n2) / 2 > Math.Min(n1, n2))
                        {
                            return "specified nu is infeasible";
                        }
                    }
                }
            }

            return null;
        }

        public static int SvmCheckProbabilityModel(Model model)
        {
            if (((model.Parameter.SvmType == SvmType.C_SVC || model.Parameter.SvmType == SvmType.NU_SVC) &&
                 model.PairwiseProbabilityA != null && model.PairwiseProbabilityB != null) ||
                ((model.Parameter.SvmType == SvmType.EPSILON_SVR || model.Parameter.SvmType == SvmType.NU_SVR) &&
                 model.PairwiseProbabilityA != null))
            {
                return 1;
            }

            return 0;
        }

        // Stratified cross validation
        public static void SvmCrossValidation(Problem problem, Parameter param, int nrFold, double[] target)
        {
            int[] perm = new int[problem.Count];

            var foldStart = GetFolds(problem, param, nrFold, perm);

            Parallel.For(
                0,
                nrFold,
                new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount / 2,
                    CancellationToken = param.Token
                },
                i =>
                {
                    int begin = foldStart[i];
                    int end = foldStart[i + 1];
                    int j, k;
                    Problem subprob = new Problem();

                    subprob.Count = problem.Count - (end - begin);
                    subprob.X = new Node[subprob.Count][];
                    subprob.Y = new double[subprob.Count];

                    k = 0;
                    for (j = 0; j < begin; j++)
                    {
                        subprob.X[k] = problem.X[perm[j]];
                        subprob.Y[k] = problem.Y[perm[j]];
                        ++k;
                    }

                    for (j = end; j < problem.Count; j++)
                    {
                        subprob.X[k] = problem.X[perm[j]];
                        subprob.Y[k] = problem.Y[perm[j]];
                        ++k;
                    }

                    ValidateFold(problem, param, target, subprob, begin, end, perm);
                });
        }

        private static void ValidateFold(Problem problem, Parameter param, double[] target, Problem subprob, int begin, int end, int[] perm)
        {
            int j;
            Model submodel = SvmTrain(subprob, param);
            if (param.Probability &&
                (param.SvmType == SvmType.C_SVC ||
                 param.SvmType == SvmType.NU_SVC))
            {
                double[] probEstimates = new double[SvmGetNrClass(submodel)];
                for (j = begin; j < end; j++)
                {
                    target[perm[j]] = SvmPredictProbability(submodel, problem.X[perm[j]], probEstimates);
                }
            }
            else
            {
                for (j = begin; j < end; j++)
                {
                    target[perm[j]] = SvmPredict(submodel, problem.X[perm[j]]);
                }
            }
        }

        private static int[] GetFolds(Problem problem, Parameter param, int nrFold, int[] perm)
        {
            int[] foldStart = new int[nrFold + 1];
            int i;

            // stratified cv may not give leave-one-out rate
            // Each class to l folds -> some folds may have zero elements
            if ((param.SvmType == SvmType.C_SVC ||
                 param.SvmType == SvmType.NU_SVC) && nrFold < problem.Count)
            {
                int[] tmpNrClass = new int[1];
                int[][] tmpLabel = new int[1][];
                int[][] tmpStart = new int[1][];
                int[][] tmpCount = new int[1][];

                SvmGroupClasses(problem, tmpNrClass, tmpLabel, tmpStart, tmpCount, perm);

                int nrClass = tmpNrClass[0];
                int[] start = tmpStart[0];
                int[] count = tmpCount[0];

                // random shuffle and then data grouped by fold using the array perm
                int[] foldCount = new int[nrFold];
                int c;
                int[] index = new int[problem.Count];
                for (i = 0; i < problem.Count; i++)
                {
                    index[i] = perm[i];
                }

                for (c = 0; c < nrClass; c++)
                {
                    for (i = 0; i < count[c]; i++)
                    {
                        int j = i + (int)(0.5 * (count[c] - i));
                        do
                        {
                            int _ = index[start[c] + j];
                            index[start[c] + j] = index[start[c] + i];
                            index[start[c] + i] = _;
                        }
                        while (false);
                    }
                }

                for (i = 0; i < nrFold; i++)
                {
                    foldCount[i] = 0;
                    for (c = 0; c < nrClass; c++)
                    {
                        foldCount[i] += (i + 1) * count[c] / nrFold - i * count[c] / nrFold;
                    }
                }

                foldStart[0] = 0;
                for (i = 1; i <= nrFold; i++)
                {
                    foldStart[i] = foldStart[i - 1] + foldCount[i - 1];
                }

                for (c = 0; c < nrClass; c++)
                {
                    for (i = 0; i < nrFold; i++)
                    {
                        int begin = start[c] + i * count[c] / nrFold;
                        int end = start[c] + (i + 1) * count[c] / nrFold;
                        for (int j = begin; j < end; j++)
                        {
                            perm[foldStart[i]] = index[j];
                            foldStart[i]++;
                        }
                    }
                }

                foldStart[0] = 0;
                for (i = 1; i <= nrFold; i++)
                {
                    foldStart[i] = foldStart[i - 1] + foldCount[i - 1];
                }
            }
            else
            {
                for (i = 0; i < problem.Count; i++)
                {
                    perm[i] = i;
                }

                for (i = 0; i < problem.Count; i++)
                {
                    int j = i + (int)(0.5 * (problem.Count - i));
                    do
                    {
                        int _ = perm[i];
                        perm[i] = perm[j];
                        perm[j] = _;
                    }
                    while (false);
                }

                for (i = 0; i <= nrFold; i++)
                {
                    foldStart[i] = i * problem.Count / nrFold;
                }
            }

            return foldStart;
        }

        public static void SvmGetLabels(Model model, int[] label)
        {
            if (model.ClassLabels != null)
            {
                for (int i = 0; i < model.NumberOfClasses; i++)
                {
                    label[i] = model.ClassLabels[i];
                }
            }
        }

        public static int SvmGetNrClass(Model model)
        {
            return model.NumberOfClasses;
        }

        public static SvmType SvmGetSvmType(Model model)
        {
            return model.Parameter.SvmType;
        }

        public static double SvmGetSvrProbability(Model model)
        {
            if ((model.Parameter.SvmType == SvmType.EPSILON_SVR || model.Parameter.SvmType == SvmType.NU_SVR) &&
                model.PairwiseProbabilityA != null)
            {
                return model.PairwiseProbabilityA[0];
            }

            log.Error("Model doesn't contain information for SVR probability inference");
            return 0;
        }

        public static double SvmPredict(Model model, Node[] x)
        {
            if (model.Parameter.SvmType == SvmType.ONE_CLASS ||
                model.Parameter.SvmType == SvmType.EPSILON_SVR ||
                model.Parameter.SvmType == SvmType.NU_SVR)
            {
                double[] res = new double[1];
                SvmPredictValues(model, x, res);

                if (model.Parameter.SvmType == SvmType.ONE_CLASS)
                {
                    return res[0] > 0 ? 1 : -1;
                }

                return res[0];
            }

            int i;
            int nrClass = model.NumberOfClasses;
            double[] decValues = new double[nrClass * (nrClass - 1) / 2];
            SvmPredictValues(model, x, decValues);

            int[] vote = new int[nrClass];
            for (i = 0; i < nrClass; i++)
            {
                vote[i] = 0;
            }

            int pos = 0;
            for (i = 0; i < nrClass; i++)
            {
                for (int j = i + 1; j < nrClass; j++)
                {
                    if (decValues[pos++] > 0)
                    {
                        ++vote[i];
                    }
                    else
                    {
                        ++vote[j];
                    }
                }
            }

            int voteMaxIdx = 0;
            for (i = 1; i < nrClass; i++)
            {
                if (vote[i] > vote[voteMaxIdx])
                {
                    voteMaxIdx = i;
                }
            }

            return model.ClassLabels[voteMaxIdx];
        }

        public static double SvmPredictProbability(Model model, Node[] x, double[] probEstimates)
        {
            if ((model.Parameter.SvmType == SvmType.C_SVC || model.Parameter.SvmType == SvmType.NU_SVC) &&
                model.PairwiseProbabilityA != null && model.PairwiseProbabilityB != null)
            {
                int i;
                int nrClass = model.NumberOfClasses;
                double[] decValues = new double[nrClass * (nrClass - 1) / 2];
                SvmPredictValues(model, x, decValues);
                const double minProb = 1e-7;
                double[,] pairwiseProb = new double[nrClass, nrClass];

                int k = 0;
                for (i = 0; i < nrClass; i++)
                {
                    for (int j = i + 1; j < nrClass; j++)
                    {
                        pairwiseProb[i, j] = Math.Min(Math.Max(SigmoidPredict(decValues[k], model.PairwiseProbabilityA[k], model.PairwiseProbabilityB[k]), minProb), 1 - minProb);
                        pairwiseProb[j, i] = 1 - pairwiseProb[i, j];
                        k++;
                    }
                }

                MulticlassProbability(nrClass, pairwiseProb, probEstimates);

                int probMaxIdx = 0;
                for (i = 1; i < nrClass; i++)
                {
                    if (probEstimates[i] > probEstimates[probMaxIdx])
                    {
                        probMaxIdx = i;
                    }
                }

                return model.ClassLabels[probMaxIdx];
            }

            return SvmPredict(model, x);
        }

        private static void SvmPredictValues(Model model, Node[] x, double[] decValues)
        {
            if (x.Length == 0)
            {
                throw new ArgumentOutOfRangeException("x");
            }

            if (model.Parameter.SvmType == SvmType.ONE_CLASS ||
                model.Parameter.SvmType == SvmType.EPSILON_SVR ||
                model.Parameter.SvmType == SvmType.NU_SVR)
            {
                double[] svCoef = model.SupportVectorCoefficients[0];
                double[] kvalue = new double[model.SupportVectorCount];
                Parallel.For(
                    0,
                    model.SupportVectorCount,
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = Environment.ProcessorCount / 2,
                        CancellationToken = model.Parameter.Token
                    },
                    i =>
                    {
                        kvalue[i] = svCoef[i] * Kernel.KernelFunction(x, model.SupportVectors[i], model.Parameter);
                    });

                var sum = kvalue.Sum();
                sum -= model.Rho[0];
                decValues[0] = sum;
            }
            else
            {
                int i;
                int nrClass = model.NumberOfClasses;

                double[] kvalue = new double[model.SupportVectorCount];
                Parallel.For(
                    0,
                    model.SupportVectorCount,
                    new ParallelOptions
                    {
                        MaxDegreeOfParallelism = Environment.ProcessorCount / 2,
                        CancellationToken = model.Parameter.Token
                    },
                    index =>
                    {
                        kvalue[index] = Kernel.KernelFunction(x, model.SupportVectors[index], model.Parameter);
                    });
                
                int[] start = new int[nrClass];
                start[0] = 0;
                for (i = 1; i < nrClass; i++)
                {
                    start[i] = start[i - 1] + model.NumberOfSVPerClass[i - 1];
                }

                int p = 0;
                for (i = 0; i < nrClass; i++)
                {
                    for (int j = i + 1; j < nrClass; j++)
                    {
                        double sum = 0;
                        int si = start[i];
                        int sj = start[j];
                        int ci = model.NumberOfSVPerClass[i];
                        int cj = model.NumberOfSVPerClass[j];

                        int k;
                        double[] coef1 = model.SupportVectorCoefficients[j - 1];
                        double[] coef2 = model.SupportVectorCoefficients[i];
                        for (k = 0; k < ci; k++)
                        {
                            var value = coef1[si + k] * kvalue[si + k];
                            if (value != 0)
                            {
                                sum += value;
                            }
                        }

                        for (k = 0; k < cj; k++)
                        {
                            var value = coef2[sj + k] * kvalue[sj + k];
                            if (value != 0)
                            {
                                sum += value;
                            }
                        }

                        sum -= model.Rho[p];
                        decValues[p] = sum;
                        p++;
                    }
                }
            }
        }

        /// <summary>
        /// Interface functions
        /// </summary>
        /// <param name="prob"></param>
        /// <param name="param"></param>
        /// <returns></returns>
        public static Model SvmTrain(Problem prob, Parameter param)
        {
            Model model = new Model
            {
                Parameter = param
            };

            if (param.SvmType == SvmType.ONE_CLASS ||
                param.SvmType == SvmType.EPSILON_SVR ||
                param.SvmType == SvmType.NU_SVR)
            {
                Regression(prob, param, model);
            }
            else
            {
                Classification(prob, param, model);
            }

            return model;
        }

        private static void Classification(Problem prob, Parameter param, Model model)
        {
            // classification
            int l = prob.Count;
            int[] tmpNrClass = new int[1];
            int[][] tmpLabel = new int[1][];
            int[][] tmpStart = new int[1][];
            int[][] tmpCount = new int[1][];
            int[] perm = new int[l];

            // group training data of the same class
            SvmGroupClasses(prob, tmpNrClass, tmpLabel, tmpStart, tmpCount, perm);
            int nrClass = tmpNrClass[0];
            int[] label = tmpLabel[0];
            int[] start = tmpStart[0];
            int[] count = tmpCount[0];
            Node[][] x = new Node[l][];
            int i;
            for (i = 0; i < l; i++)
            {
                x[i] = prob.X[perm[i]];
            }

            // calculate weighted C
            double[] weightedC = new double[nrClass];
            for (i = 0; i < nrClass; i++)
            {
                weightedC[i] = param.C;
            }

            foreach (int weightedLabel in param.Weights.Keys)
            {
                int index = Array.IndexOf(label, weightedLabel);
                if (index < 0)
                {
                    log.Error("warning: class label " + weightedLabel + " specified in weight is not found");
                }
                else
                {
                    weightedC[index] *= param.Weights[weightedLabel];
                }
            }

            // train k*(k-1)/2 models

            bool[] nonzero = new bool[l];
            for (i = 0; i < l; i++)
            {
                nonzero[i] = false;
            }

            DecisionFunction[] f = new DecisionFunction[nrClass * (nrClass - 1) / 2];
            double[] probA = null, probB = null;
            if (param.Probability)
            {
                probA = new double[nrClass * (nrClass - 1) / 2];
                probB = new double[nrClass * (nrClass - 1) / 2];
            }

            int p = 0;
            for (i = 0; i < nrClass; i++)
            {
                for (int j = i + 1; j < nrClass; j++)
                {
                    param.Token.ThrowIfCancellationRequested();
                    Problem subProb = new Problem();
                    int si = start[i], sj = start[j];
                    int ci = count[i], cj = count[j];
                    subProb.Count = ci + cj;
                    subProb.X = new Node[subProb.Count][];
                    subProb.Y = new double[subProb.Count];
                    int k;
                    for (k = 0; k < ci; k++)
                    {
                        subProb.X[k] = x[si + k];
                        subProb.Y[k] = +1;
                    }

                    for (k = 0; k < cj; k++)
                    {
                        subProb.X[ci + k] = x[sj + k];
                        subProb.Y[ci + k] = -1;
                    }

                    if (param.Probability)
                    {
                        double[] probAB = new double[2];
                        SvmBinarySvcProbability(subProb, param, weightedC[i], weightedC[j], probAB);
                        probA[p] = probAB[0];
                        probB[p] = probAB[1];
                    }

                    f[p] = SvmTrainOne(subProb, param, weightedC[i], weightedC[j]);
                    for (k = 0; k < ci; k++)
                    {
                        if (!nonzero[si + k] && Math.Abs(f[p].Alpha[k]) > 0)
                        {
                            nonzero[si + k] = true;
                        }
                    }

                    for (k = 0; k < cj; k++)
                    {
                        if (!nonzero[sj + k] && Math.Abs(f[p].Alpha[ci + k]) > 0)
                        {
                            nonzero[sj + k] = true;
                        }
                    }

                    ++p;
                }
            }

            // build output
            model.NumberOfClasses = nrClass;
            model.ClassLabels = new int[nrClass];
            for (i = 0; i < nrClass; i++)
            {
                model.ClassLabels[i] = label[i];
            }

            model.Rho = new double[nrClass * (nrClass - 1) / 2];
            for (i = 0; i < nrClass * (nrClass - 1) / 2; i++)
            {
                model.Rho[i] = f[i].Rho;
            }

            if (param.Probability)
            {
                model.PairwiseProbabilityA = new double[nrClass * (nrClass - 1) / 2];
                model.PairwiseProbabilityB = new double[nrClass * (nrClass - 1) / 2];
                for (i = 0; i < nrClass * (nrClass - 1) / 2; i++)
                {
                    model.PairwiseProbabilityA[i] = probA[i];
                    model.PairwiseProbabilityB[i] = probB[i];
                }
            }
            else
            {
                model.PairwiseProbabilityA = null;
                model.PairwiseProbabilityB = null;
            }

            int nnz = 0;
            int[] nzCount = new int[nrClass];
            model.NumberOfSVPerClass = new int[nrClass];
            for (i = 0; i < nrClass; i++)
            {
                int nSV = 0;
                for (int j = 0; j < count[i]; j++)
                {
                    if (!nonzero[start[i] + j])
                    {
                        continue;
                    }
                    ++nSV;
                    ++nnz;
                }

                model.NumberOfSVPerClass[i] = nSV;
                nzCount[i] = nSV;
            }

            log.Debug("Total nSV = " + nnz);
            model.SupportVectorCount = nnz;
            model.SupportVectors = new Node[nnz][];
            p = 0;
            for (i = 0; i < l; i++)
            {
                if (nonzero[i])
                {
                    model.SupportVectors[p++] = x[i];
                }
            }

            int[] nzStart = new int[nrClass];
            nzStart[0] = 0;
            for (i = 1; i < nrClass; i++)
            {
                nzStart[i] = nzStart[i - 1] + nzCount[i - 1];
            }

            model.SupportVectorCoefficients = new double[nrClass - 1][];
            for (i = 0; i < nrClass - 1; i++)
            {
                model.SupportVectorCoefficients[i] = new double[nnz];
            }

            p = 0;
            for (i = 0; i < nrClass; i++)
            {
                for (int j = i + 1; j < nrClass; j++)
                {
                    // classifier (i,j): coefficients with
                    // i are in sv_coef[j-1][nz_start[i]...],
                    // j are in sv_coef[i][nz_start[j]...]

                    int si = start[i];
                    int sj = start[j];
                    int ci = count[i];
                    int cj = count[j];

                    int q = nzStart[i];
                    int k;
                    for (k = 0; k < ci; k++)
                    {
                        if (nonzero[si + k])
                        {
                            model.SupportVectorCoefficients[j - 1][q++] = f[p].Alpha[k];
                        }
                    }

                    q = nzStart[j];
                    for (k = 0; k < cj; k++)
                    {
                        if (nonzero[sj + k])
                        {
                            model.SupportVectorCoefficients[i][q++] = f[p].Alpha[ci + k];
                        }
                    }

                    ++p;
                }
            }
        }

        private static void Regression(Problem prob, Parameter param, Model model)
        {
            // regression or one-class-svm
            model.NumberOfClasses = 2;
            model.ClassLabels = null;
            model.NumberOfSVPerClass = null;
            model.PairwiseProbabilityA = null;
            model.PairwiseProbabilityB = null;
            model.SupportVectorCoefficients = new double[1][];

            if (param.Probability &&
                (param.SvmType == SvmType.EPSILON_SVR ||
                 param.SvmType == SvmType.NU_SVR))
            {
                model.PairwiseProbabilityA = new double[1];
                model.PairwiseProbabilityA[0] = SvmSvrProbability(prob, param);
            }

            DecisionFunction f = SvmTrainOne(prob, param, 0, 0);
            model.Rho = new double[1];
            model.Rho[0] = f.Rho;

            int nSv = 0;
            int i;
            for (i = 0; i < prob.Count; i++)
            {
                if (Math.Abs(f.Alpha[i]) > 0)
                {
                    ++nSv;
                }
            }

            model.SupportVectorCount = nSv;
            model.SupportVectors = new Node[nSv][];
            model.SupportVectorCoefficients[0] = new double[nSv];
            int j = 0;
            for (i = 0; i < prob.Count; i++)
            {
                if (Math.Abs(f.Alpha[i]) > 0)
                {
                    model.SupportVectors[j] = prob.X[i];
                    model.SupportVectorCoefficients[0][j] = f.Alpha[i];
                    ++j;
                }
            }
        }

        // Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
        private static void MulticlassProbability(int k, double[,] r, double[] p)
        {
            int t, j;
            int iter = 0, maxIter = Math.Max(100, k);
            double[,] Q = new double[k, k];
            double[] Qp = new double[k];
            double pQp, eps = 0.005 / k;

            for (t = 0; t < k; t++)
            {
                p[t] = 1.0 / k; // Valid if k = 1
                Q[t, t] = 0;
                for (j = 0; j < t; j++)
                {
                    Q[t, t] += r[j, t] * r[j, t];
                    Q[t, j] = Q[j, t];
                }
                for (j = t + 1; j < k; j++)
                {
                    Q[t, t] += r[j, t] * r[j, t];
                    Q[t, j] = -r[j, t] * r[t, j];
                }
            }

            for (iter = 0; iter < maxIter; iter++)
            {
                // stopping condition, recalculate QP,pQP for numerical accuracy
                pQp = 0;
                for (t = 0; t < k; t++)
                {
                    Qp[t] = 0;
                    for (j = 0; j < k; j++)
                    {
                        Qp[t] += Q[t, j] * p[j];
                    }
                    pQp += p[t] * Qp[t];
                }

                double maxError = 0;
                for (t = 0; t < k; t++)
                {
                    double error = Math.Abs(Qp[t] - pQp);
                    if (error > maxError)
                    {
                        maxError = error;
                    }
                }
                if (maxError < eps)
                {
                    break;
                }

                for (t = 0; t < k; t++)
                {
                    double diff = (-Qp[t] + pQp) / Q[t, t];
                    p[t] += diff;
                    pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
                    for (j = 0; j < k; j++)
                    {
                        Qp[j] = (Qp[j] + diff * Q[t, j]) / (1 + diff);
                        p[j] /= 1 + diff;
                    }
                }
            }

            if (iter >= maxIter)
            {
                log.Info("Exceeds Max_iter in multiclass_prob");
            }
        }

        private static double SigmoidPredict(double decisionValue, double A, double B)
        {
            double fApB = decisionValue * A + B;
            if (fApB >= 0)
            {
                return Math.Exp(-fApB) / (1.0 + Math.Exp(-fApB));
            }

            return 1.0 / (1 + Math.Exp(fApB));
        }

        // Platt's binary SVM Probablistic Output: an improvement from Lin et al.
        private static void SigmoidTrain(int l, double[] decValues, double[] labels, double[] probAb)
        {
            double a, b;
            double prior1 = 0, prior0 = 0;
            int i;

            for (i = 0; i < l; i++)
            {
                if (labels[i] > 0)
                {
                    prior1 += 1;
                }
                else
                {
                    prior0 += 1;
                }
            }

            const int maxIter = 100;
            const double minStep = 1e-10;
            const double sigma = 1e-12;
            const double eps = 1e-5;
            double hiTarget = (prior1 + 1.0) / (prior1 + 2.0);
            double loTarget = 1 / (prior0 + 2.0);
            double[] t = new double[l];
            double fApB, p, q, h11, h22, h21, g1, g2, det, dA, dB, gd, stepsize;
            double newA, newB, newf, d1, d2;
            int iter;

            // Initial Point and Initial Fun value
            a = 0.0;
            b = Math.Log((prior0 + 1.0) / (prior1 + 1.0));
            double fval = 0.0;

            for (i = 0; i < l; i++)
            {
                if (labels[i] > 0)
                {
                    t[i] = hiTarget;
                }
                else
                {
                    t[i] = loTarget;
                }

                fApB = decValues[i] * a + b;
                if (fApB >= 0)
                {
                    fval += t[i] * fApB + Math.Log(1 + Math.Exp(-fApB));
                }
                else
                {
                    fval += (t[i] - 1) * fApB + Math.Log(1 + Math.Exp(fApB));
                }
            }
            for (iter = 0; iter < maxIter; iter++)
            {
                // Update Gradient and Hessian (use H' = H + sigma I)
                h11 = sigma; // numerically ensures strict PD
                h22 = sigma;
                h21 = 0.0;
                g1 = 0.0;
                g2 = 0.0;
                for (i = 0; i < l; i++)
                {
                    fApB = decValues[i] * a + b;
                    if (fApB >= 0)
                    {
                        p = Math.Exp(-fApB) / (1.0 + Math.Exp(-fApB));
                        q = 1.0 / (1.0 + Math.Exp(-fApB));
                    }
                    else
                    {
                        p = 1.0 / (1.0 + Math.Exp(fApB));
                        q = Math.Exp(fApB) / (1.0 + Math.Exp(fApB));
                    }
                    d2 = p * q;
                    h11 += decValues[i] * decValues[i] * d2;
                    h22 += d2;
                    h21 += decValues[i] * d2;
                    d1 = t[i] - p;
                    g1 += decValues[i] * d1;
                    g2 += d1;
                }

                // Stopping Criteria
                if (Math.Abs(g1) < eps && Math.Abs(g2) < eps)
                {
                    break;
                }

                // Finding Newton direction: -inv(H') * g
                det = h11 * h22 - h21 * h21;
                dA = -(h22 * g1 - h21 * g2) / det;
                dB = -(-h21 * g1 + h11 * g2) / det;
                gd = g1 * dA + g2 * dB;

                stepsize = 1; // Line Search
                while (stepsize >= minStep)
                {
                    newA = a + stepsize * dA;
                    newB = b + stepsize * dB;

                    // New function value
                    newf = 0.0;
                    for (i = 0; i < l; i++)
                    {
                        fApB = decValues[i] * newA + newB;
                        if (fApB >= 0)
                        {
                            newf += t[i] * fApB + Math.Log(1 + Math.Exp(-fApB));
                        }
                        else
                        {
                            newf += (t[i] - 1) * fApB + Math.Log(1 + Math.Exp(fApB));
                        }
                    }

                    // Check sufficient decrease
                    if (newf < fval + 0.0001 * stepsize * gd)
                    {
                        a = newA;
                        b = newB;
                        fval = newf;
                        break;
                    }

                    stepsize = stepsize / 2.0;
                }

                if (stepsize < minStep)
                {
                    log.Info("Line search fails in two-class probability estimates");
                    break;
                }
            }

            if (iter >= maxIter)
            {
                log.Info("Reaching Maximal iterations in two-class probability estimates");
            }

            probAb[0] = a;
            probAb[1] = b;
        }

        private static void SolveCSvc(
            Problem prob,
            Parameter param,
            double[] alpha,
            SolutionInfo si,
            double cp,
            double cn)
        {
            int l = prob.Count;
            double[] minusOnes = new double[l];
            sbyte[] y = new sbyte[l];

            int i;

            for (i = 0; i < l; i++)
            {
                alpha[i] = 0;
                minusOnes[i] = -1;
                if (prob.Y[i] > 0)
                {
                    y[i] = +1;
                }
                else
                {
                    y[i] = -1;
                }
            }

            Solver solver = new Solver();
            solver.Token = param.Token;
            solver.Solve(
                l,
                new SvcQ(prob, param, y),
                minusOnes,
                y,
                alpha,
                cp,
                cn,
                param.EPS,
                si,
                param.Shrinking);

            double sumAlpha = 0;
            for (i = 0; i < l; i++)
            {
                sumAlpha += alpha[i];
            }

            if (cp == cn)
            {
                log.Debug("nu = " + sumAlpha / (cp * prob.Count));
            }

            for (i = 0; i < l; i++)
            {
                alpha[i] *= y[i];
            }
        }

        private static void SolveEpsilonSvr(Problem prob, Parameter param, double[] alpha, SolutionInfo si)
        {
            int l = prob.Count;
            double[] alpha2 = new double[2 * l];
            double[] linearTerm = new double[2 * l];
            sbyte[] y = new sbyte[2 * l];
            int i;

            for (i = 0; i < l; i++)
            {
                alpha2[i] = 0;
                linearTerm[i] = param.P - prob.Y[i];
                y[i] = 1;

                alpha2[i + l] = 0;
                linearTerm[i + l] = param.P + prob.Y[i];
                y[i + l] = -1;
            }

            Solver solver = new Solver();
            solver.Token = param.Token;
            solver.Solve(2 * l, new SvrQ(prob, param), linearTerm, y, alpha2, param.C, param.C, param.EPS, si, param.Shrinking);

            double sumAlpha = 0;
            for (i = 0; i < l; i++)
            {
                alpha[i] = alpha2[i] - alpha2[i + l];
                sumAlpha += Math.Abs(alpha[i]);
            }

            log.Debug("nu = " + sumAlpha / (param.C * l));
        }

        private static void SolveNuSvc(
            Problem problem,
            Parameter parameter,
            double[] alpha,
            SolutionInfo solutionInfo)
        {
            int i;
            int totalProblems = problem.Count;
            double nu = parameter.Nu;

            sbyte[] y = new sbyte[totalProblems];

            for (i = 0; i < totalProblems; i++)
            {
                if (problem.Y[i] > 0)
                {
                    y[i] = +1;
                }
                else
                {
                    y[i] = -1;
                }
            }

            double sumPositive = nu * totalProblems / 2;
            double sumNegative = nu * totalProblems / 2;

            for (i = 0; i < totalProblems; i++)
            {
                if (y[i] == +1)
                {
                    alpha[i] = Math.Min(1.0, sumPositive);
                    sumPositive -= alpha[i];
                }
                else
                {
                    alpha[i] = Math.Min(1.0, sumNegative);
                    sumNegative -= alpha[i];
                }
            }

            double[] zeros = new double[totalProblems];

            for (i = 0; i < totalProblems; i++)
            {
                zeros[i] = 0;
            }

            SolverNu solver = new SolverNu();
            solver.Token = parameter.Token;
            solver.Solve(totalProblems, new SvcQ(problem, parameter, y), zeros, y, alpha, 1.0, 1.0, parameter.EPS, solutionInfo, parameter.Shrinking);
            double r = solutionInfo.R;

            log.Debug("C = " + 1 / r);

            for (i = 0; i < totalProblems; i++)
            {
                alpha[i] *= y[i] / r;
            }

            solutionInfo.Rho /= r;
            solutionInfo.Obj /= r * r;
            solutionInfo.UpperBoundP = 1 / r;
            solutionInfo.UpperBoundN = 1 / r;
        }

        private static void SolveNuSvr(
            Problem prob,
            Parameter param,
            double[] alpha,
            SolutionInfo si)
        {
            int l = prob.Count;
            double C = param.C;
            double[] alpha2 = new double[2 * l];
            double[] linear_term = new double[2 * l];
            sbyte[] y = new sbyte[2 * l];
            int i;

            double sum = C * param.Nu * l / 2;
            for (i = 0; i < l; i++)
            {
                alpha2[i] = alpha2[i + l] = Math.Min(sum, C);
                sum -= alpha2[i];

                linear_term[i] = -prob.Y[i];
                y[i] = 1;

                linear_term[i + l] = prob.Y[i];
                y[i + l] = -1;
            }

            SolverNu solver = new SolverNu();
            solver.Token = param.Token;
            solver.Solve(2 * l, new SvrQ(prob, param), linear_term, y, alpha2, C, C, param.EPS, si, param.Shrinking);

            log.Debug("epsilon = " + -si.R);

            for (i = 0; i < l; i++)
            {
                alpha[i] = alpha2[i] - alpha2[i + l];
            }
        }

        private static void SolveOneClass(
            Problem prob,
            Parameter param,
            double[] alpha,
            SolutionInfo si)
        {
            int l = prob.Count;
            double[] zeros = new double[l];
            sbyte[] ones = new sbyte[l];
            int i;

            int n = (int)(param.Nu * prob.Count); // # of Alpha's at upper bound

            for (i = 0; i < n; i++)
            {
                alpha[i] = 1;
            }

            if (n < prob.Count)
            {
                alpha[n] = param.Nu * prob.Count - n;
            }

            for (i = n + 1; i < l; i++)
            {
                alpha[i] = 0;
            }

            for (i = 0; i < l; i++)
            {
                zeros[i] = 0;
                ones[i] = 1;
            }

            Solver solver = new Solver();
            solver.Token = param.Token;
            solver.Solve(l, new OneClassQ(prob, param), zeros, ones, alpha, 1.0, 1.0, param.EPS, si, param.Shrinking);
        }

        /// <summary>
        /// Cross-validation decision values for probability estimates
        /// </summary>
        /// <param name="problem"></param>
        /// <param name="param"></param>
        /// <param name="cp"></param>
        /// <param name="cn"></param>
        /// <param name="probAb"></param>
        private static void SvmBinarySvcProbability(Problem problem, Parameter param, double cp, double cn, double[] probAb)
        {
            int i;
            const int nrFold = 5;
            int[] perm = new int[problem.Count];
            double[] decValues = new double[problem.Count];

            // random shuffle
            Random rand = new Random();
            for (i = 0; i < problem.Count; i++)
            {
                perm[i] = i;
            }

            for (i = 0; i < problem.Count; i++)
            {
                int j = i + (int)(rand.NextDouble() * (problem.Count - i));
                do
                {
                    int _ = perm[i];
                    perm[i] = perm[j];
                    perm[j] = _;
                }
                while (false);
            }

            for (i = 0; i < nrFold; i++)
            {
                int begin = i * problem.Count / nrFold;
                int end = (i + 1) * problem.Count / nrFold;
                int j, k;
                Problem subprob = new Problem();

                subprob.Count = problem.Count - (end - begin);
                subprob.X = new Node[subprob.Count][];
                subprob.Y = new double[subprob.Count];

                k = 0;
                for (j = 0; j < begin; j++)
                {
                    subprob.X[k] = problem.X[perm[j]];
                    subprob.Y[k] = problem.Y[perm[j]];
                    ++k;
                }
                for (j = end; j < problem.Count; j++)
                {
                    subprob.X[k] = problem.X[perm[j]];
                    subprob.Y[k] = problem.Y[perm[j]];
                    ++k;
                }

                int pCount = 0, nCount = 0;
                for (j = 0; j < k; j++)
                {
                    if (subprob.Y[j] > 0)
                    {
                        pCount++;
                    }
                    else
                    {
                        nCount++;
                    }
                }

                if (pCount == 0 && nCount == 0)
                {
                    for (j = begin; j < end; j++)
                    {
                        decValues[perm[j]] = 0;
                    }
                }
                else if (pCount > 0 && nCount == 0)
                {
                    for (j = begin; j < end; j++)
                    {
                        decValues[perm[j]] = 1;
                    }
                }
                else if (pCount == 0 && nCount > 0)
                {
                    for (j = begin; j < end; j++)
                    {
                        decValues[perm[j]] = -1;
                    }
                }
                else
                {
                    Parameter subparam = (Parameter)param.Clone();
                    subparam.Probability = false;
                    subparam.C = 1.0;
                    subparam.Weights[1] = cp;
                    subparam.Weights[-1] = cn;
                    Model submodel = SvmTrain(subprob, subparam);
                    for (j = begin; j < end; j++)
                    {
                        double[] decValue = new double[1];
                        SvmPredictValues(submodel, problem.X[perm[j]], decValue);
                        decValues[perm[j]] = decValue[0];

                        // ensure +1 -1 order; reason not using CV subroutine
                        decValues[perm[j]] *= submodel.ClassLabels[0];
                    }
                }
            }

            SigmoidTrain(problem.Count, decValues, problem.Y, probAb);
        }

        // label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
        // perm, length l, must be allocated before calling this subroutine
        private static void SvmGroupClasses(Problem prob, int[] nrClassRet, int[][] labelRet, int[][] startRet, int[][] countRet, int[] perm)
        {
            int l = prob.Count;
            int maxNrClass = 16;
            int nrClass = 0;
            int[] label = new int[maxNrClass];
            int[] count = new int[maxNrClass];
            int[] dataLabel = new int[l];
            int i;

            for (i = 0; i < l; i++)
            {
                int thisLabel = (int)prob.Y[i];
                int j;
                for (j = 0; j < nrClass; j++)
                {
                    if (thisLabel == label[j])
                    {
                        ++count[j];
                        break;
                    }
                }

                dataLabel[i] = j;
                if (j == nrClass)
                {
                    if (nrClass == maxNrClass)
                    {
                        maxNrClass *= 2;
                        int[] newData = new int[maxNrClass];
                        Array.Copy(label, 0, newData, 0, label.Length);
                        label = newData;
                        newData = new int[maxNrClass];
                        Array.Copy(count, 0, newData, 0, count.Length);
                        count = newData;
                    }

                    label[nrClass] = thisLabel;
                    count[nrClass] = 1;
                    ++nrClass;
                }
            }

            int[] start = new int[nrClass];
            start[0] = 0;
            for (i = 1; i < nrClass; i++)
            {
                start[i] = start[i - 1] + count[i - 1];
            }

            for (i = 0; i < l; i++)
            {
                perm[start[dataLabel[i]]] = i;
                ++start[dataLabel[i]];
            }

            start[0] = 0;
            for (i = 1; i < nrClass; i++)
            {
                start[i] = start[i - 1] + count[i - 1];
            }

            nrClassRet[0] = nrClass;
            labelRet[0] = label;
            startRet[0] = start;
            countRet[0] = count;
        }

        // Return parameter of a Laplace distribution 
        private static double SvmSvrProbability(Problem problem, Parameter param)
        {
            int i;
            const int nrFold = 5;
            double[] ymv = new double[problem.Count];
            double mae = 0;

            Parameter newparam = (Parameter)param.Clone();
            newparam.Probability = false;
            SvmCrossValidation(problem, newparam, nrFold, ymv);
            for (i = 0; i < problem.Count; i++)
            {
                ymv[i] = problem.Y[i] - ymv[i];
                mae += Math.Abs(ymv[i]);
            }

            mae /= problem.Count;
            double std = Math.Sqrt(2 * mae * mae);
            int count = 0;
            mae = 0;
            for (i = 0; i < problem.Count; i++)
            {
                if (Math.Abs(ymv[i]) > 5 * std)
                {
                    count = count + 1;
                }
                else
                {
                    mae += Math.Abs(ymv[i]);
                }
            }

            mae /= problem.Count - count;
            log.Debug("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=" + mae + "\n");
            return mae;
        }

        private static DecisionFunction SvmTrainOne(Problem prob, Parameter param, double Cp, double Cn)
        {
            double[] alpha = new double[prob.Count];
            SolutionInfo si = new SolutionInfo();
            switch (param.SvmType)
            {
                case SvmType.C_SVC:
                    SolveCSvc(prob, param, alpha, si, Cp, Cn);
                    break;
                case SvmType.NU_SVC:
                    SolveNuSvc(prob, param, alpha, si);
                    break;
                case SvmType.ONE_CLASS:
                    SolveOneClass(prob, param, alpha, si);
                    break;
                case SvmType.EPSILON_SVR:
                    SolveEpsilonSvr(prob, param, alpha, si);
                    break;
                case SvmType.NU_SVR:
                    SolveNuSvr(prob, param, alpha, si);
                    break;
            }

            log.Debug("obj = " + si.Obj + ", rho = " + si.Rho);

            // output SVs

            int nSV = 0;
            int nBSV = 0;
            for (int i = 0; i < prob.Count; i++)
            {
                if (Math.Abs(alpha[i]) > 0)
                {
                    ++nSV;
                    if (prob.Y[i] > 0)
                    {
                        if (Math.Abs(alpha[i]) >= si.UpperBoundP)
                        {
                            ++nBSV;
                        }
                    }
                    else
                    {
                        if (Math.Abs(alpha[i]) >= si.UpperBoundN)
                        {
                            ++nBSV;
                        }
                    }
                }
            }

            log.Debug("nSV = " + nSV + ", nBSV = " + nBSV);

            DecisionFunction f = new DecisionFunction();
            f.Alpha = alpha;
            f.Rho = si.Rho;
            return f;
        }
    }
}
