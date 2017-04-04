using System.Globalization;
using System.Threading;

namespace Wikiled.MachineLearning.Svm.Logic
{
    internal static class TemporaryCulture
    {
        private static CultureInfo culture;

        public static void Start()
        {
            culture = Thread.CurrentThread.CurrentCulture;
            Thread.CurrentThread.CurrentCulture = CultureInfo.InvariantCulture;
        }

        public static void Stop()
        {
            Thread.CurrentThread.CurrentCulture = culture;
        }
    }
}
