using System;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    public class SearchProgressEventArgs : EventArgs
    {
        public SearchProgressEventArgs(int totalSteps, int currentStep, bool isNewMaximum, double maximum)
        {
            TotalSteps = totalSteps;
            CurrentStep = currentStep;
            IsNewMaximum = isNewMaximum;
            Maximum = maximum;
        }

        public int CurrentStep { get; }

        public bool IsNewMaximum { get; }

        public double Maximum { get; }

        public int TotalSteps { get; }
    }
}
