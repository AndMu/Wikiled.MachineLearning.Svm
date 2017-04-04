using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class OneClassQ : Kernel
    {
        private readonly Cache cache;

        private readonly float[] qd;

        public OneClassQ(Problem prob, Parameter param)
            : base(prob.Count, prob.X, param)
        {
            cache = new Cache(prob.Count, (long)(param.CacheSize * (1 << 20)));
            qd = new float[prob.Count];
            for (int i = 0; i < prob.Count; i++)
            {
                qd[i] = (float)KernelFunction(i, i);
            }
        }

        public sealed override float[] GetQ(int i, int len)
        {
            float[] data = null;
            int start;
            if ((start = cache.GetData(i, ref data, len)) < len)
            {
                for (int j = start; j < len; j++)
                {
                    data[j] = (float)KernelFunction(i, j);
                }
            }

            return data;
        }

        public sealed override float[] GetQD()
        {
            return qd;
        }

        public sealed override void SwapIndex(int i, int j)
        {
            cache.SwapIndex(i, j);
            base.SwapIndex(i, j);
            qd.SwapIndex(i, j);
        }
    }
}