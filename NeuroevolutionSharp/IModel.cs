namespace NeuroevolutionSharp;

public interface IModel<T>
{
    static abstract T GetZero();

    static abstract T GetNormal(double mean, double std);

    static abstract T Operate(T[] models, Func<double[], double> operateFunc);
}
