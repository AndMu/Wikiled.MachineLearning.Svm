namespace Wikiled.MachineLearning.Svm.Logic
{
    internal interface IQMatrix
    {
        float[] GetQ(int column, int len);

        float[] GetQD();

        void SwapIndex(int i, int j);
    }
}