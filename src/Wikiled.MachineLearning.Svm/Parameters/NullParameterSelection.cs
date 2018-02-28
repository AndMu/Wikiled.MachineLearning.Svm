using System.Threading;
using System.Threading.Tasks;
using Wikiled.Common.Arguments;
using Wikiled.MachineLearning.Svm.Logic;

namespace Wikiled.MachineLearning.Svm.Parameters
{
    public class NullParameterSelection : IParameterSelection
    {
        private readonly Parameter parameter;

        public NullParameterSelection(Parameter parameter, ITrainingModel training)
        {
            Guard.NotNull(() => parameter, parameter);
            Guard.NotNull(() => training, training);
            this.parameter = parameter;
            Training = training;
        }

        public ITrainingModel Training { get; }

        public GridSearchParameters SearchParameters { get; }

        public Task<Parameter> Find(Problem problem, CancellationToken token)
        {
            Guard.NotNull(() => problem, problem);
            return Task.FromResult(parameter);
        }
    }
}
