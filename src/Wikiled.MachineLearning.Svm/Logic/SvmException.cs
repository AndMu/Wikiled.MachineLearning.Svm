using System;
using System.Runtime.Serialization;

namespace Wikiled.MachineLearning.Svm.Logic
{
    [Serializable]
    public class SvmException : Exception
    {
        public SvmException()
        {
        }

        public SvmException(string message) : base(message)
        {
        }

        public SvmException(string message, Exception inner) : base(message, inner)
        {
        }

        protected SvmException(
            SerializationInfo info,
            StreamingContext context) : base(info, context)
        {
        }
    }
}
