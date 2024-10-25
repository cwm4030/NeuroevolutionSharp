namespace NeuroevolutionSharp.Models;

public interface IModel<T>
{
    static abstract T Operate(T[] models, Func<double[], double> operateFunc);
}
