namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    ///     Contains the various kernel types this library can use.
    /// </summary>
    public enum KernelType
    {
        /// <summary>
        ///     Linear: u'*v
        /// </summary>
        Linear,

        /// <summary>
        ///     Polynomial: (gamma*u'*v + coef0)^degree
        /// </summary>
        Polynomial,

        /// <summary>
        ///     Radial basis function: exp(-gamma*|u-v|^2)
        /// </summary>
        RBF,

        /// <summary>
        ///     Sigmoid: tanh(gamma*u'*v + coef0)
        /// </summary>
        Sigmoid,

        /// <summary>
        ///     Precomputed kernel
        /// </summary>
        Precomputed
    }
}
