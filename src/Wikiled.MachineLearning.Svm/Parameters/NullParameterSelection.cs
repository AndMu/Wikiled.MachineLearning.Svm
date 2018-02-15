using System.Threading;
using System.Threading.Tasks;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    public class NullParameterSelection : IParameterSelection
    {
        private readonly Parameter parameter;

        public NullParameterSelection(Parameter parameter)
        {
            Guard.NotNull(() => parameter, parameter);
            this.parameter = parameter;
        }

        public Task<Parameter> Find(Problem problem, CancellationToken token)
        {
            Guard.NotNull(() => problem, problem);
            return Task.FromResult(parameter);
        }
    }
}
