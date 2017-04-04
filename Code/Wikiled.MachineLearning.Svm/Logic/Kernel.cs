using System;
using Wikiled.Arff.Data;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal abstract class Kernel : IQMatrix
    {
        private readonly double coef0;

        private readonly int degree;

        private readonly double gamma;

        private readonly KernelType kernelType;

        private readonly Node[][] xNodes;

        private readonly double[] xSquare;

        protected Kernel(int l, Node[][] x, Parameter param)
        {
            kernelType = param.KernelType;
            degree = param.Degree;
            gamma = param.Gamma;
            coef0 = param.Coefficient0;

            xNodes = (Node[][])x.Clone();

            if (kernelType == KernelType.RBF)
            {
                xSquare = new double[l];
                for (int i = 0; i < l; i++)
                {
                    xSquare[i] = Dot(xNodes[i], xNodes[i]);
                }
            }
            else
            {
                xSquare = null;
            }
        }

        public static double KernelFunction(Node[] x, Node[] y, Parameter param)
        {
            switch (param.KernelType)
            {
                case KernelType.Linear:
                    return Dot(x, y);
                case KernelType.Polynomial:
                    return Powi(param.Degree * Dot(x, y) + param.Coefficient0, param.Degree);
                case KernelType.RBF:
                {
                    double sum = ComputeSquaredDistance(x, y);
                    return Math.Exp(-param.Gamma * sum);
                }
                case KernelType.Sigmoid:
                    return Math.Tanh(param.Gamma * Dot(x, y) + param.Coefficient0);
                case KernelType.Precomputed:
                    return x[(int)y[0].Value].Value;
                default:
                    return 0;
            }
        }

        public abstract float[] GetQ(int column, int len);

        public abstract float[] GetQD();

        public virtual void SwapIndex(int i, int j)
        {
            xNodes.SwapIndex(i, j);
            xSquare?.SwapIndex(i, j);
        }

        protected double KernelFunction(int i, int j)
        {
            switch (kernelType)
            {
                case KernelType.Linear:
                    return Dot(xNodes[i], xNodes[j]);
                case KernelType.Polynomial:
                    return Powi(gamma * Dot(xNodes[i], xNodes[j]) + coef0, degree);
                case KernelType.RBF:
                    return Math.Exp(-gamma * (xSquare[i] + xSquare[j] - 2 * Dot(xNodes[i], xNodes[j])));
                case KernelType.Sigmoid:
                    return Math.Tanh(gamma * Dot(xNodes[i], xNodes[j]) + coef0);
                case KernelType.Precomputed:
                    return xNodes[i][(int)xNodes[j][0].Value].Value;
                default:
                    return 0;
            }
        }

        private static double ComputeSquaredDistance(Node[] xNodes, Node[] yNodes)
        {
            Node x = xNodes[0];
            Node y = yNodes[0];
            int xLength = xNodes.Length;
            int yLength = yNodes.Length;
            int xIndex = 0;
            int yIndex = 0;
            double sum = 0;

            while (true)
            {
                if (x.Index == y.Index)
                {
                    double d = x.Value - y.Value;
                    sum += d * d;
                    xIndex++;
                    yIndex++;
                    if (xIndex < xLength && yIndex < yLength)
                    {
                        x = xNodes[xIndex];
                        y = yNodes[yIndex];
                    }
                    else if (xIndex < xLength)
                    {
                        x = xNodes[xIndex];
                        break;
                    }
                    else if (yIndex < yLength)
                    {
                        y = yNodes[yIndex];
                        break;
                    }
                    else
                    {
                        break;
                    }
                }
                else if (x.Index > y.Index)
                {
                    sum += y.Value * y.Value;
                    if (++yIndex < yLength)
                    {
                        y = yNodes[yIndex];
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    sum += x.Value * x.Value;
                    if (++xIndex < xLength)
                    {
                        x = xNodes[xIndex];
                    }
                    else
                    {
                        break;
                    }
                }
            }

            for (; xIndex < xLength; xIndex++)
            {
                double d = xNodes[xIndex].Value;
                sum += d * d;
            }

            for (; yIndex < yLength; yIndex++)
            {
                double d = yNodes[yIndex].Value;
                sum += d * d;
            }

            return sum;
        }

        private static double Dot(Node[] xNodes, Node[] yNodes)
        {
            double sum = 0;
            int xlen = xNodes.Length;
            int ylen = yNodes.Length;
            if (xlen == 0)
            {
                throw new ArgumentOutOfRangeException("xNodes");
            }

            if (ylen == 0)
            {
                throw new ArgumentOutOfRangeException("yNodes");
            }

            int i = 0;
            int j = 0;
            Node x = xNodes[0];
            Node y = yNodes[0];
            while (true)
            {
                if (x.Index == y.Index)
                {
                    sum += (x.Value * y.Value);
                    i++;
                    j++;
                    if (i < xlen && j < ylen)
                    {
                        x = xNodes[i];
                        y = yNodes[j];
                    }
                    else
                    {
                        break;
                    }
                }
                else
                {
                    if (x.Index > y.Index)
                    {
                        ++j;
                        if (j < ylen)
                        {
                            y = yNodes[j];
                        }
                        else
                        {
                            break;
                        }
                    }
                    else
                    {
                        ++i;
                        if (i < xlen)
                        {
                            x = xNodes[i];
                        }
                        else
                        {
                            break;
                        }
                    }
                }
            }

            return sum;
        }

        private static double Powi(double value, int times)
        {
            double tmp = value, ret = 1.0;
            for (int i = times; i > 0; i /= 2)
            {
                if (i % 2 == 1)
                {
                    ret *= tmp;
                }

                tmp = tmp * tmp;
            }

            return ret;
        }
    }
}
