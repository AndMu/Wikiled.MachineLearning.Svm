using System;

namespace Wikiled.MachineLearning.Svm.Logic
{
    //
    // Solver for nu-svm classification and regression
    //
    // additional constraint: e^T \Alpha = constant
    //
    class SolverNu : Solver
    {
        private SolutionInfo si;

        public sealed override void Solve(int l, IQMatrix matrix, double[] p, sbyte[] y,
               double[] alpha, double Cp, double Cn, double eps,
               SolutionInfo si, bool shrinking)
        {
            this.si = si;
            base.Solve(l, matrix, p, y, alpha, Cp, Cn, eps, si, shrinking);
        }

        // return 1 if already optimal, return 0 otherwise
        protected override int SelectWorkingSet(int[] workingSet)
        {
            // return i,j such that y_i = y_j and
            // i: Maximizes -y_i * grad(f)_i, i in I_up(\Alpha)
            // j: Minimizes the decrease of obj value
            //    (if quadratic coefficeint <= 0, replace it with tau)
            //    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\Alpha)

            double GMaxp = -INF;
            double GMaxp2 = -INF;
            int GMaxp_idx = -1;

            double GMaxn = -INF;
            double GMaxn2 = -INF;
            int GMaxn_idx = -1;

            int GMin_idx = -1;
            double obj_diff_Min = INF;

            for (int t = 0; t < activeSize; t++)
            {
                if (y[t] == +1)
                {
                    if (!IsUpperBound(t))
                    {
                        if (-gradient[t] >= GMaxp)
                        {
                            GMaxp = -gradient[t];
                            GMaxp_idx = t;
                        }
                    }
                }
                else
                {
                    if (!IsLowerBound(t))
                    {
                        if (gradient[t] >= GMaxn)
                        {
                            GMaxn = gradient[t];
                            GMaxn_idx = t;
                        }
                    }
                }
            }
            int ip = GMaxp_idx;
            int iN = GMaxn_idx;
            float[] Q_ip = null;
            float[] Q_in = null;
            if (ip != -1) // null Q_ip not accessed: GMaxp=-INF if ip=-1
            {
                Q_ip = qMatrix.GetQ(ip, activeSize);
            }
            if (iN != -1)
            {
                Q_in = qMatrix.GetQ(iN, activeSize);
            }

            for (int j = 0; j < activeSize; j++)
            {
                if (y[j] == +1)
                {
                    if (!IsLowerBound(j))
                    {
                        double grad_diff = GMaxp + gradient[j];
                        if (gradient[j] >= GMaxp2)
                            GMaxp2 = gradient[j];
                        if (grad_diff > 0)
                        {
                            double obj_diff;
                            double quad_coef = Q_ip[ip] + qd[j] - 2 * Q_ip[j];
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = -(grad_diff * grad_diff) / 1e-12;

                            if (obj_diff <= obj_diff_Min)
                            {
                                GMin_idx = j;
                                obj_diff_Min = obj_diff;
                            }
                        }
                    }
                }
                else
                {
                    if (!IsUpperBound(j))
                    {
                        double grad_diff = GMaxn - gradient[j];
                        if (-gradient[j] >= GMaxn2)
                            GMaxn2 = -gradient[j];
                        if (grad_diff > 0)
                        {
                            double obj_diff;
                            double quad_coef = Q_in[iN] + qd[j] - 2 * Q_in[j];
                            if (quad_coef > 0)
                                obj_diff = -(grad_diff * grad_diff) / quad_coef;
                            else
                                obj_diff = -(grad_diff * grad_diff) / 1e-12;

                            if (obj_diff <= obj_diff_Min)
                            {
                                GMin_idx = j;
                                obj_diff_Min = obj_diff;
                            }
                        }
                    }
                }
            }

            if (Math.Max(GMaxp + GMaxp2, GMaxn + GMaxn2) < eps)
                return 1;

            if (y[GMin_idx] == +1)
                workingSet[0] = GMaxp_idx;
            else
                workingSet[0] = GMaxn_idx;
            workingSet[1] = GMin_idx;

            return 0;
        }

        private bool BeShrunk(int i, double gMax1, double gMax2, double gMax3, double gMax4)
        {
            if (IsUpperBound(i))
            {
                if (y[i] == +1)
                    return (-gradient[i] > gMax1);
                return (-gradient[i] > gMax4);
            }
            if (IsLowerBound(i))
            {
                if (y[i] == +1)
                    return (gradient[i] > gMax2);
                return (gradient[i] > gMax3);
            }
            return (false);
        }

        protected override void DoShrinking()
        {
            double gMax1 = -INF;	// Max { -y_i * grad(f)_i | y_i = +1, i in I_up(\Alpha) }
            double gMax2 = -INF;	// Max { y_i * grad(f)_i | y_i = +1, i in I_low(\Alpha) }
            double gMax3 = -INF;	// Max { -y_i * grad(f)_i | y_i = -1, i in I_up(\Alpha) }
            double gMax4 = -INF;	// Max { y_i * grad(f)_i | y_i = -1, i in I_low(\Alpha) }

            // find Maximal violating pair first
            int i;
            for (i = 0; i < activeSize; i++)
            {
                if (!IsUpperBound(i))
                {
                    if (y[i] == +1)
                    {
                        if (-gradient[i] > gMax1) gMax1 = -gradient[i];
                    }
                    else if (-gradient[i] > gMax4) gMax4 = -gradient[i];
                }
                if (!IsLowerBound(i))
                {
                    if (y[i] == +1)
                    {
                        if (gradient[i] > gMax2) gMax2 = gradient[i];
                    }
                    else if (gradient[i] > gMax3) gMax3 = gradient[i];
                }
            }

            if (unshrink == false && Math.Max(gMax1 + gMax2, gMax3 + gMax4) <= eps * 10)
            {
                unshrink = true;
                ReconstructGradient();
                activeSize = totalProblems;
            }

            for (i = 0; i < activeSize; i++)
            {
                if (BeShrunk(i, gMax1, gMax2, gMax3, gMax4))
                {
                    activeSize--;
                    while (activeSize > i)
                    {
                        if (!BeShrunk(activeSize, gMax1, gMax2, gMax3, gMax4))
                        {
                            SwapIndex(i, activeSize);
                            break;
                        }
                        activeSize--;
                    }
                }
            }
        }

        protected override double CalculateRho()
        {
            int nr_free1 = 0, nr_free2 = 0;
            double ub1 = INF, ub2 = INF;
            double lb1 = -INF, lb2 = -INF;
            double sum_free1 = 0, sum_free2 = 0;

            for (int i = 0; i < activeSize; i++)
            {
                if (y[i] == +1)
                {
                    if (IsLowerBound(i))
                        ub1 = Math.Min(ub1, gradient[i]);
                    else if (IsUpperBound(i))
                        lb1 = Math.Max(lb1, gradient[i]);
                    else
                    {
                        ++nr_free1;
                        sum_free1 += gradient[i];
                    }
                }
                else
                {
                    if (IsLowerBound(i))
                        ub2 = Math.Min(ub2, gradient[i]);
                    else if (IsUpperBound(i))
                        lb2 = Math.Max(lb2, gradient[i]);
                    else
                    {
                        ++nr_free2;
                        sum_free2 += gradient[i];
                    }
                }
            }

            double r1, r2;
            if (nr_free1 > 0)
                r1 = sum_free1 / nr_free1;
            else
                r1 = (ub1 + lb1) / 2;

            if (nr_free2 > 0)
                r2 = sum_free2 / nr_free2;
            else
                r2 = (ub2 + lb2) / 2;

            si.R = (r1 + r2) / 2;
            return (r1 - r2) / 2;
        }
    }
}
