﻿namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    /// Contains all of the types of SVM this library can model.
    /// </summary>
    public enum SvmType { 
        /// <summary>
        /// C-SVC.
        /// </summary>
        C_SVC, 

        /// <summary>
        /// nu-SVC.
        /// </summary>
        NU_SVC, 

        /// <summary>
        /// one-class SVM
        /// </summary>
        ONE_CLASS, 

        /// <summary>
        /// epsilon-SVR
        /// </summary>
        EPSILON_SVR, 

        /// <summary>
        /// nu-SVR
        /// </summary>
        NU_SVR 
    };
}