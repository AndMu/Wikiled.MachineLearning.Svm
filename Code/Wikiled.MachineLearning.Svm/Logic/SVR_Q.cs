using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class SvrQ : Kernel
    {
        private readonly float[][] buffer;

        private readonly Cache cache;

        private readonly int[] index;

        private readonly int l;

        private readonly float[] qd;

        private readonly sbyte[] sign;

        private int nextBuffer;

        public SvrQ(Problem prob, Parameter param)
            : base(prob.Count, prob.X, param)
        {
            l = prob.Count;
            cache = new Cache(l, (long)(param.CacheSize * (1 << 20)));
            qd = new float[2 * l];
            sign = new sbyte[2 * l];
            index = new int[2 * l];
            for (int k = 0; k < l; k++)
            {
                sign[k] = 1;
                sign[k + l] = -1;
                index[k] = k;
                index[k + l] = k;
                qd[k] = (float)KernelFunction(k, k);
                qd[k + l] = qd[k];
            }

            buffer = new float[2][];
            buffer[0] = new float[2 * l];
            buffer[1] = new float[2 * l];
            nextBuffer = 0;
        }

        public sealed override float[] GetQ(int i, int len)
        {
            float[] data = null;
            int realI = index[i];
            if (cache.GetData(realI, ref data, l) < l)
            {
                for (int j = 0; j < l; j++)
                {
                    data[j] = (float)KernelFunction(realI, j);
                }
            }

            // reorder and copy
            float[] buf = buffer[nextBuffer];
            nextBuffer = 1 - nextBuffer;
            sbyte si = sign[i];
            for (int j = 0; j < len; j++)
            {
                buf[j] = (float)si * sign[j] * data[index[j]];
            }

            return buf;
        }

        public sealed override float[] GetQD()
        {
            return qd;
        }

        public sealed override void SwapIndex(int i, int j)
        {
            sign.SwapIndex(i, j);
            index.SwapIndex(i, j);
            qd.SwapIndex(i, j);
        }
    }
}
