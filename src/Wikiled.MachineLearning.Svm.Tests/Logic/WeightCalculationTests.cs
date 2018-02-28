using System;
using NUnit.Framework;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Tests.Logic
{
    [TestFixture]
    public class WeightCalculationTests
    {
        [TestCase(new[] { 1, 1, 1, 0 }, new[] { 1, 0.33, 0, 1 })]
        [TestCase(new[] { 1, 0 }, new double[] { 1, 1, 0, 1 })]
        [TestCase(new[] { 1, 1, 1, 0, 2 }, new[] { 1, 0.33, 0, 1, 2, 1 })]
        [TestCase(new[] { 1, 1, 1, 0, 2, 2 }, new[] { 1, 0.33, 0, 1, 2, 0.5 })]
        public void GetWeights(int[] values, double[] result)
        {
            var weights = WeightCalculation.GetWeights(values);
            for (int i = 0; i < result.Length; i = i + 2)
            {
                Assert.AreEqual(result[i + 1], Math.Round(weights[(int)result[i]], 2));
            }
        }
    }
}
