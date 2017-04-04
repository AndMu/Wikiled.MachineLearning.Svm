using System;
using System.Threading;
using NLog;

namespace Wikiled.MachineLearning.Svm.Logic
{
    // An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
    // Solves:
    //
    //	Min 0.5(\Alpha^T Q \Alpha) + p^T \Alpha
    //
    //		y^T \Alpha = \delta
    //		y_i = +1 or -1
    //		0 <= alpha_i <= Cp for y_i = 1
    //		0 <= alpha_i <= Cn for y_i = -1
    //
    // Given:
    //
    //	Q, p, y, Cp, Cn, and an initial feasible point \Alpha
    //	l is the size of vectors and matrices
    //	eps is the stopping tolerance
    //
    // solution will be put in \Alpha, objective value will be put in obj
    //
    internal class Solver
    {
        protected const double INF = double.PositiveInfinity;

        private const byte FREE = 2;

        private const byte LOWER_BOUND = 0;

        private const byte UPPER_BOUND = 1;

        private static readonly Logger log = LogManager.GetCurrentClassLogger();

        protected int activeSize;

        protected double eps;

        protected double[] gradient; // gradient of objective function

        protected float[] qd;

        protected IQMatrix qMatrix;

        protected int totalProblems;

        protected bool unshrink; // XXX

        protected sbyte[] y;

        private int[] activeSet;

        private double[] alpha;

        private byte[] alphaStatus; // LOWER_BOUND, UPPER_BOUND, FREE

        private double cn;

        private double cp;

        private double[] gradientBar; // gradient, if we treat free variables as 0

        private double[] p;

        public CancellationToken Token { get; set; }

        public virtual void Solve(
            int totalProblemsInSolver,
            IQMatrix matrix,
            double[] pValue,
            sbyte[] yValue,
            double[] alphaValue,
            double cpValue,
            double cnValue,
            double epsValue,
            SolutionInfo solutionInfo,
            bool shrinking)
        {
            totalProblems = totalProblemsInSolver;
            qMatrix = matrix;
            qd = matrix.GetQD();
            p = (double[])pValue.Clone();
            y = (sbyte[])yValue.Clone();
            alpha = (double[])alphaValue.Clone();
            this.cp = cpValue;
            this.cn = cnValue;
            eps = epsValue;
            unshrink = false;

            // initialize alpha_status
            InitializeAlphaStatus(totalProblemsInSolver);

            // initialize active set (for shrinking)
            InitializeActiveSet(totalProblemsInSolver);

            // initialize gradient
            InitializeGradient(totalProblemsInSolver, matrix);

            // optimization step

            int maxIterations = Math.Max(10000000, totalProblemsInSolver > int.MaxValue / 100 ? int.MaxValue : 100 * totalProblemsInSolver);
            int iter = 0;
            int counter = Math.Min(totalProblemsInSolver, 1000) + 1;
            int[] workingSet = new int[2];

            while (iter < maxIterations)
            {
                // show progress and do shrinking
                Token.ThrowIfCancellationRequested();
                if (--counter == 0)
                {
                    counter = Math.Min(totalProblemsInSolver, 1000);
                    if (shrinking)
                    {
                        DoShrinking();
                    }

                    log.Debug(".");
                }

                if (SelectWorkingSet(workingSet) != 0)
                {
                    // reconstruct the whole gradient
                    ReconstructGradient();

                    // reset active set size and check
                    activeSize = totalProblemsInSolver;
                    log.Debug("*");
                    if (SelectWorkingSet(workingSet) != 0)
                    {
                        break;
                    }
                    counter = 1; // do shrinking next iteration
                }

                int i = workingSet[0];
                int j = workingSet[1];

                ++iter;

                // update Alpha[i] and Alpha[j], handle bounds carefully

                float[] qI = matrix.GetQ(i, activeSize);
                float[] qJ = matrix.GetQ(j, activeSize);

                double cI = GetC(i);
                double cJ = GetC(j);

                double oldAlphaI = alpha[i];
                double oldAlphaJ = alpha[j];

                if (y[i] != y[j])
                {
                    double quadCoef = qI[i] + qJ[j] + 2 * qI[j];
                    if (quadCoef <= 0)
                    {
                        quadCoef = 1e-12;
                    }

                    double delta = (-gradient[i] - gradient[j]) / quadCoef;
                    double diff = alpha[i] - alpha[j];
                    alpha[i] += delta;
                    alpha[j] += delta;

                    if (diff > 0)
                    {
                        if (alpha[j] < 0)
                        {
                            alpha[j] = 0;
                            alpha[i] = diff;
                        }
                    }
                    else
                    {
                        if (alpha[i] < 0)
                        {
                            alpha[i] = 0;
                            alpha[j] = -diff;
                        }
                    }

                    if (diff > cI - cJ)
                    {
                        if (alpha[i] > cI)
                        {
                            alpha[i] = cI;
                            alpha[j] = cI - diff;
                        }
                    }
                    else
                    {
                        if (alpha[j] > cJ)
                        {
                            alpha[j] = cJ;
                            alpha[i] = cJ + diff;
                        }
                    }
                }
                else
                {
                    double quadCoef = qI[i] + qJ[j] - 2 * qI[j];
                    if (quadCoef <= 0)
                    {
                        quadCoef = 1e-12;
                    }

                    double delta = (gradient[i] - gradient[j]) / quadCoef;
                    double sum = alpha[i] + alpha[j];
                    alpha[i] -= delta;
                    alpha[j] += delta;

                    if (sum > cI)
                    {
                        if (alpha[i] > cI)
                        {
                            alpha[i] = cI;
                            alpha[j] = sum - cI;
                        }
                    }
                    else
                    {
                        if (alpha[j] < 0)
                        {
                            alpha[j] = 0;
                            alpha[i] = sum;
                        }
                    }
                    if (sum > cJ)
                    {
                        if (alpha[j] > cJ)
                        {
                            alpha[j] = cJ;
                            alpha[i] = sum - cJ;
                        }
                    }
                    else
                    {
                        if (alpha[i] < 0)
                        {
                            alpha[i] = 0;
                            alpha[j] = sum;
                        }
                    }
                }

                // update G
                double deltaAlphaI = alpha[i] - oldAlphaI;
                double deltaAlphaJ = alpha[j] - oldAlphaJ;

                for (int k = 0; k < activeSize; k++)
                {
                    gradient[k] += qI[k] * deltaAlphaI + qJ[k] * deltaAlphaJ;
                }

                // update alpha_status and G_bar

                {
                    bool ui = IsUpperBound(i);
                    bool uj = IsUpperBound(j);
                    UpdateAlphaStatus(i);
                    UpdateAlphaStatus(j);
                    int k;
                    if (ui != IsUpperBound(i))
                    {
                        qI = matrix.GetQ(i, totalProblemsInSolver);
                        if (ui)
                        {
                            for (k = 0; k < totalProblemsInSolver; k++)
                            {
                                gradientBar[k] -= cI * qI[k];
                            }
                        }
                        else
                        {
                            for (k = 0; k < totalProblemsInSolver; k++)
                            {
                                gradientBar[k] += cI * qI[k];
                            }
                        }
                    }

                    if (uj != IsUpperBound(j))
                    {
                        qJ = matrix.GetQ(j, totalProblemsInSolver);
                        if (uj)
                        {
                            for (k = 0; k < totalProblemsInSolver; k++)
                            {
                                gradientBar[k] -= cJ * qJ[k];
                            }
                        }
                        else
                        {
                            for (k = 0; k < totalProblemsInSolver; k++)
                            {
                                gradientBar[k] += cJ * qJ[k];
                            }
                        }
                    }
                }
            }

            if (iter >= maxIterations)
            {
                if (activeSize < totalProblemsInSolver)
                {
                    // reconstruct the whole gradient to calculate objective value
                    ReconstructGradient();
                    activeSize = totalProblemsInSolver;
                    log.Debug("*");
                }

                log.Warn("WARNING: reaching max number of iterations");
            }

            // calculate rho
            solutionInfo.Rho = CalculateRho();

            // calculate objective value
            CalculateObjectiveValue(totalProblemsInSolver, solutionInfo);

            // put back the solution
            for (int i = 0; i < totalProblemsInSolver; i++)
            {
                alphaValue[activeSet[i]] = alpha[i];
            }

            solutionInfo.UpperBoundP = cpValue;
            solutionInfo.UpperBoundN = cnValue;
            log.Debug("optimization finished, #iter = " + iter);
        }

        protected virtual double CalculateRho()
        {
            double r;
            int nr_free = 0;
            double ub = INF, lb = -INF, sum_free = 0;
            for (int i = 0; i < activeSize; i++)
            {
                double yG = y[i] * gradient[i];

                if (IsLowerBound(i))
                {
                    if (y[i] > 0)
                    {
                        ub = Math.Min(ub, yG);
                    }
                    else
                    {
                        lb = Math.Max(lb, yG);
                    }
                }
                else if (IsUpperBound(i))
                {
                    if (y[i] < 0)
                    {
                        ub = Math.Min(ub, yG);
                    }
                    else
                    {
                        lb = Math.Max(lb, yG);
                    }
                }
                else
                {
                    ++nr_free;
                    sum_free += yG;
                }
            }

            if (nr_free > 0)
            {
                r = sum_free / nr_free;
            }
            else
            {
                r = (ub + lb) / 2;
            }

            return r;
        }

        protected virtual void DoShrinking()
        {
            int i;
            double GMax1 = -INF; // Max { -y_i * grad(f)_i | i in I_up(\Alpha) }
            double GMax2 = -INF; // Max { y_i * grad(f)_i | i in I_low(\Alpha) }

            // find Maximal violating pair first
            for (i = 0; i < activeSize; i++)
            {
                if (y[i] == +1)
                {
                    if (!IsUpperBound(i))
                    {
                        if (-gradient[i] >= GMax1)
                        {
                            GMax1 = -gradient[i];
                        }
                    }
                    if (!IsLowerBound(i))
                    {
                        if (gradient[i] >= GMax2)
                        {
                            GMax2 = gradient[i];
                        }
                    }
                }
                else
                {
                    if (!IsUpperBound(i))
                    {
                        if (-gradient[i] >= GMax2)
                        {
                            GMax2 = -gradient[i];
                        }
                    }
                    if (!IsLowerBound(i))
                    {
                        if (gradient[i] >= GMax1)
                        {
                            GMax1 = gradient[i];
                        }
                    }
                }
            }

            if (unshrink == false && GMax1 + GMax2 <= eps * 10)
            {
                unshrink = true;
                ReconstructGradient();
                activeSize = totalProblems;
            }

            for (i = 0; i < activeSize; i++)
            {
                if (BeShrunk(i, GMax1, GMax2))
                {
                    activeSize--;
                    while (activeSize > i)
                    {
                        if (!BeShrunk(activeSize, GMax1, GMax2))
                        {
                            SwapIndex(i, activeSize);
                            break;
                        }
                        activeSize--;
                    }
                }
            }
        }

        // return 1 if already optimal, return 0 otherwise
        protected virtual int SelectWorkingSet(int[] workingSet)
        {
            // return i,j such that
            // i: Maximizes -y_i * grad(f)_i, i in I_up(\Alpha)
            // j: mimimizes the decrease of obj value
            //    (if quadratic coefficeint <= 0, replace it with tau)
            //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\Alpha)

            double gMax = -INF;
            double gMax2 = -INF;
            int gMaxIdx = -1;
            int gMinIdx = -1;
            double objDiffMin = INF;

            for (int t = 0; t < activeSize; t++)
            {
                if (y[t] == +1)
                {
                    if (!IsUpperBound(t))
                    {
                        if (-gradient[t] >= gMax)
                        {
                            gMax = -gradient[t];
                            gMaxIdx = t;
                        }
                    }
                }
                else
                {
                    if (!IsLowerBound(t))
                    {
                        if (gradient[t] >= gMax)
                        {
                            gMax = gradient[t];
                            gMaxIdx = t;
                        }
                    }
                }
            }
            int i = gMaxIdx;
            float[] qI = null;
            if (i != -1) // null Q_i not accessed: GMax=-INF if i=-1
            {
                qI = qMatrix.GetQ(i, activeSize);
            }
            for (int j = 0; j < activeSize; j++)
            {
                if (y[j] == +1)
                {
                    if (!IsLowerBound(j))
                    {
                        double gradDiff = gMax + gradient[j];
                        if (gradient[j] >= gMax2)
                        {
                            gMax2 = gradient[j];
                        }
                        if (gradDiff > 0)
                        {
                            double objDiff;
                            double quadCoef = qI[i] + qd[j] - 2.0 * y[i] * qI[j];
                            if (quadCoef > 0)
                            {
                                objDiff = -(gradDiff * gradDiff) / quadCoef;
                            }
                            else
                            {
                                objDiff = -(gradDiff * gradDiff) / 1e-12;
                            }
                            if (objDiff <= objDiffMin)
                            {
                                gMinIdx = j;
                                objDiffMin = objDiff;
                            }
                        }
                    }
                }
                else
                {
                    if (!IsUpperBound(j))
                    {
                        double gradDiff = gMax - gradient[j];
                        if (-gradient[j] >= gMax2)
                        {
                            gMax2 = -gradient[j];
                        }
                        if (gradDiff > 0)
                        {
                            double objDiff;
                            double quadCoef = qI[i] + qd[j] + 2.0 * y[i] * qI[j];
                            if (quadCoef > 0)
                            {
                                objDiff = -(gradDiff * gradDiff) / quadCoef;
                            }
                            else
                            {
                                objDiff = -(gradDiff * gradDiff) / 1e-12;
                            }

                            if (objDiff <= objDiffMin)
                            {
                                gMinIdx = j;
                                objDiffMin = objDiff;
                            }
                        }
                    }
                }
            }

            if (gMax + gMax2 < eps)
            {
                return 1;
            }

            workingSet[0] = gMaxIdx;
            workingSet[1] = gMinIdx;
            return 0;
        }

        protected bool IsLowerBound(int i)
        {
            return alphaStatus[i] == LOWER_BOUND;
        }

        protected bool IsUpperBound(int i)
        {
            return alphaStatus[i] == UPPER_BOUND;
        }

        protected void ReconstructGradient()
        {
            // reconstruct inactive elements of G from G_bar and free variables
            if (activeSize == totalProblems)
            {
                return;
            }

            int i, j;
            int nrFree = 0;

            for (j = activeSize; j < totalProblems; j++)
            {
                gradient[j] = gradientBar[j] + p[j];
            }

            for (j = 0; j < activeSize; j++)
            {
                if (IsFree(j))
                {
                    nrFree++;
                }
            }

            if (2 * nrFree < activeSize)
            {
                log.Warn("Warning: using -h 0 may be faster");
            }

            if (nrFree * totalProblems > 2 * activeSize * (totalProblems - activeSize))
            {
                for (i = activeSize; i < totalProblems; i++)
                {
                    float[] qI = qMatrix.GetQ(i, activeSize);
                    for (j = 0; j < activeSize; j++)
                    {
                        if (IsFree(j))
                        {
                            gradient[i] += alpha[j] * qI[j];
                        }
                    }
                }
            }
            else
            {
                for (i = 0; i < activeSize; i++)
                {
                    if (IsFree(i))
                    {
                        float[] qI = qMatrix.GetQ(i, totalProblems);
                        double alphaI = alpha[i];
                        for (j = activeSize; j < totalProblems; j++)
                        {
                            gradient[j] += alphaI * qI[j];
                        }
                    }
                }
            }
        }

        protected void SwapIndex(int i, int j)
        {
            qMatrix.SwapIndex(i, j);
            y.SwapIndex(i, j);
            gradient.SwapIndex(i, j);
            alphaStatus.SwapIndex(i, j);
            alpha.SwapIndex(i, j);
            p.SwapIndex(i, j);
            activeSet.SwapIndex(i, j);
            gradientBar.SwapIndex(i, j);
        }

        private bool BeShrunk(int i, double gMax1, double gMax2)
        {
            if (IsUpperBound(i))
            {
                if (y[i] == +1)
                {
                    return -gradient[i] > gMax1;
                }
                return -gradient[i] > gMax2;
            }
            if (IsLowerBound(i))
            {
                if (y[i] == +1)
                {
                    return gradient[i] > gMax2;
                }
                return gradient[i] > gMax1;
            }
            return false;
        }

        private void CalculateObjectiveValue(int totalProblems, SolutionInfo solutionInfo)
        {
            double v = 0;
            int i;
            for (i = 0; i < totalProblems; i++)
            {
                v += alpha[i] * (gradient[i] + p[i]);
            }

            solutionInfo.Obj = v / 2;
        }

        private double GetC(int i)
        {
            return y[i] > 0 ? cp : cn;
        }

        private void InitializeActiveSet(int totalProblems)
        {
            activeSet = new int[totalProblems];
            for (int i = 0; i < totalProblems; i++)
            {
                activeSet[i] = i;
            }
            activeSize = totalProblems;
        }

        private void InitializeAlphaStatus(int totalProblems)
        {
            alphaStatus = new byte[totalProblems];
            for (int i = 0; i < totalProblems; i++)
            {
                UpdateAlphaStatus(i);
            }
        }

        private void InitializeGradient(int totalProblems, IQMatrix qMatrix)
        {
            gradient = new double[totalProblems];
            gradientBar = new double[totalProblems];
            int i;
            for (i = 0; i < totalProblems; i++)
            {
                gradient[i] = p[i];
                gradientBar[i] = 0;
            }
            for (i = 0; i < totalProblems; i++)
            {
                if (!IsLowerBound(i))
                {
                    float[] Q_i = qMatrix.GetQ(i, totalProblems);
                    double alpha_i = alpha[i];
                    int j;
                    for (j = 0; j < totalProblems; j++)
                    {
                        gradient[j] += alpha_i * Q_i[j];
                    }
                    if (IsUpperBound(i))
                    {
                        for (j = 0; j < totalProblems; j++)
                        {
                            gradientBar[j] += GetC(i) * Q_i[j];
                        }
                    }
                }
            }
        }

        private bool IsFree(int i)
        {
            return alphaStatus[i] == FREE;
        }

        private void UpdateAlphaStatus(int i)
        {
            if (alpha[i] >= GetC(i))
            {
                alphaStatus[i] = UPPER_BOUND;
            }
            else if (alpha[i] <= 0)
            {
                alphaStatus[i] = LOWER_BOUND;
            }
            else
            {
                alphaStatus[i] = FREE;
            }
        }
    }
}
