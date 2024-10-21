namespace NeuroevolutionSharp;

public class AdamOptimizer<T>(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    where T : IModel<T>
{
    private readonly double _learningRate = learningRate;

    private readonly double _beta1 = beta1;

    private readonly double _beta2 = beta2;

    private readonly double _epsilon = epsilon;

    private Func<double[], double> _updateFunc = x => x[0] - x[1];

    private T _m = T.GetZero();

    private T _v = T.GetZero();

    private ulong _t = 0;

    public AdamOptimizer<T> GradientAscent()
    {
        _updateFunc = x => x[0] + x[1];
        return this;
    }

    public AdamOptimizer<T> GradientDescent()
    {
        _updateFunc = x => x[0] - x[1];
        return this;
    }

    public T Update(T model, T gradient)
    {
        _t += 1;
        _m = T.Operate([_m, gradient], x => _beta1 * x[0] + (1 - _beta1) * x[1]);
        _v = T.Operate([_v, gradient], x => _beta2 * x[0] + (1 - _beta2) * x[1] * x[1]);
        var mHat = T.Operate([_m], x => x[0] / (1 - Math.Pow(_beta1, _t)));
        var vHat = T.Operate([_v], x => x[0] / (1 - Math.Pow(_beta2, _t)));
        var update = T.Operate([mHat, vHat], x => _learningRate * x[0] / (Math.Sqrt(x[1]) + _epsilon));
        return T.Operate([model, update], _updateFunc);
    }
}
