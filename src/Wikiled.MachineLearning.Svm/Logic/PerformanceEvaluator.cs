using System.Collections.Generic;

namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    ///     Class which evaluates an SVM model using several standard techniques.
    /// </summary>
    public class PerformanceEvaluator
    {
        private readonly PredictionResult result;

        private List<ChangePoint> changes;

        private List<RankPair> data;

        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="set">A pre-computed ranked pair set</param>
        public PerformanceEvaluator(List<RankPair> set)
        {
            data = set;
            computeStatistics();
        }

        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="model">Model to evaluate</param>
        /// <param name="problem">Problem to evaluate</param>
        /// <param name="category">Category to evaluate for</param>
        public PerformanceEvaluator(Model model, Problem problem, double category)
        {
            result = Prediction.Predict(problem, model, true);
            ParseResults(problem.Y, category);
            computeStatistics();
        }

        /// <summary>
        ///     Constructor.
        /// </summary>
        /// <param name="resultsFile">Results file</param>
        /// <param name="correctLabels">The correct labels of each data item</param>
        /// <param name="category">The category to evaluate for</param>
        public PerformanceEvaluator(string resultsFile, int[] correctLabels, double category)
        {
            ParseResults(correctLabels, category);
            computeStatistics();
        }

        /// <summary>
        ///     The average precision
        /// </summary>
        public double AP { get; private set; }

        /// <summary>
        ///     Returns the area under the ROC Curve
        /// </summary>
        public double AuC { get; private set; }

        /// <summary>
        ///     Precision-Recall curve
        /// </summary>
        public List<CurvePoint> PRCurve { get; private set; }

        /// <summary>
        ///     Receiver Operating Characteristic curve
        /// </summary>
        public List<CurvePoint> ROCCurve { get; private set; }

        private static float ComputeFpr(ChangePoint cp)
        {
            return (float)cp.FP / (cp.FP + cp.TN);
        }

        private static float ComputePrecision(ChangePoint p)
        {
            return (float)p.TP / (p.TP + p.FP);
        }

        private static float ComputeRecall(ChangePoint p)
        {
            return (float)p.TP / (p.TP + p.FN);
        }

        private static float ComputeTpr(ChangePoint cp)
        {
            return ComputeRecall(cp);
        }

        private void ComputePR()
        {
            PRCurve = new List<CurvePoint>();
            PRCurve.Add(new CurvePoint(0, 1));
            float precision = ComputePrecision(changes[0]);
            float recall = ComputeRecall(changes[0]);
            float precisionSum = 0;
            if (changes[0].TP > 0)
            {
                precisionSum += precision;
                PRCurve.Add(new CurvePoint(recall, precision));
            }
            for (int i = 1; i < changes.Count; i++)
            {
                precision = ComputePrecision(changes[i]);
                recall = ComputeRecall(changes[i]);
                if (changes[i].TP > changes[i - 1].TP)
                {
                    precisionSum += precision;
                    PRCurve.Add(new CurvePoint(recall, precision));
                }
            }

            PRCurve.Add(new CurvePoint(1, (float)(changes[0].TP + changes[0].FN) / (changes[0].FP + changes[0].TN)));
            AP = precisionSum / (changes[0].FN + changes[0].TP);
        }

        private void ComputeRoC()
        {
            ROCCurve = new List<CurvePoint>();
            ROCCurve.Add(new CurvePoint(0, 0));
            float tpr = ComputeTpr(changes[0]);
            float fpr = ComputeFpr(changes[0]);
            ROCCurve.Add(new CurvePoint(fpr, tpr));
            AuC = 0;
            for (int i = 1; i < changes.Count; i++)
            {
                float newTPR = ComputeTpr(changes[i]);
                float newFPR = ComputeFpr(changes[i]);
                if (changes[i].TP > changes[i - 1].TP)
                {
                    AuC += tpr * (newFPR - fpr) + .5 * (newTPR - tpr) * (newFPR - fpr);
                    tpr = newTPR;
                    fpr = newFPR;
                    ROCCurve.Add(new CurvePoint(fpr, tpr));
                }
            }

            ROCCurve.Add(new CurvePoint(1, 1));
            AuC += tpr * (1 - fpr) + .5 * (1 - tpr) * (1 - fpr);
        }

        private void computeStatistics()
        {
            data.Sort();

            findChanges();
            ComputePR();
            ComputeRoC();
        }

        private void findChanges()
        {
            int tp, fp, tn, fn;
            tp = fp = tn = fn = 0;
            for (int i = 0; i < data.Count; i++)
            {
                if (data[i].Label == 1)
                {
                    fn++;
                }
                else
                {
                    tn++;
                }
            }

            changes = new List<ChangePoint>();
            for (int i = 0; i < data.Count; i++)
            {
                if (data[i].Label == 1)
                {
                    tp++;
                    fn--;
                }
                else
                {
                    fp++;
                    tn--;
                }

                changes.Add(new ChangePoint(tp, fp, tn, fn));
            }
        }

        private void ParseResults(int[] labels, double category)
        {
            int confidenceIndex = -1;
            var retrievedLabels = result.Labels;
            for (int i = 0; i < retrievedLabels.Length; i++)
            {
                if (retrievedLabels[i] == category)
                {
                    confidenceIndex = i;
                    break;
                }
            }
            data = new List<RankPair>();
            for (int i = 0; i < labels.Length; i++)
            {
                var classItem = result.Classes[i];
                double confidence = classItem.Values[confidenceIndex];
                data.Add(new RankPair(confidence, labels[i] == category ? 1 : 0));
            }
        }
    }
}
