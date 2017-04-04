namespace Wikiled.MachineLearning.Svm.Logic
{
    internal sealed class HeadT
    {
        internal float[] data;

        internal int len; // data[0,len) is cached in this entry

        internal HeadT prev, next; // a cicular list

        public HeadT(Cache enclosingInstance)
        {
            EnclosingInstance = enclosingInstance;
        }

        public Cache EnclosingInstance { get; }
    }
}