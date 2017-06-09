using System.Threading.Tasks;
using Wikiled.MachineLearning.Svm.Extensions;
using Wikiled.MachineLearning.Svm.Parameters;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal class SvcQ : Kernel
    {
        private readonly sbyte[] y;

        private readonly Cache cache;

        private readonly float[] qd;

        public SvcQ(Problem prob, Parameter param, sbyte[] y_)
            : base(prob.Count, prob.X, param)
        {
            y = (sbyte[])y_.Clone();
            cache = new Cache(prob.Count, (long)(param.CacheSize * (1 << 20)));
            qd = new float[prob.Count];
            Parallel.For(
                0,
                prob.Count,
                new ParallelOptions
                {
                    CancellationToken = param.Token
                },
                i =>
                {
                    qd[i] = (float)KernelFunction(i, i);
                });
        }

        public sealed override float[] GetQ(int index, int len)
        {
            float[] data = null;
            int start;
            if ((start = cache.GetData(index, ref data, len)) < len)
            {
                Parallel.For(
                    start,
                    len,
                    new ParallelOptions
                    {
                        CancellationToken = Param.Token
                    },
                    i =>
                    {
                        data[i] = (float)(y[index] * y[i] * KernelFunction(index, i));
                    });
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
            y.SwapIndex(i, j);
            qd.SwapIndex(i, j);
        }
    }
}
