namespace NeuroevolutionSharp;

// uses the polar form of the Box-Muller transform to get a normal distribution
public static class NormalDistribution
{
    private static readonly Random s_random = new();

    private static double? _z2 = null;

    public static double GetSample(double mean, double std)
    {
        if (_z2 != null)
        {
            var z2 = _z2 ?? 0;
            _z2 = null;
            return std * z2 + mean;
        }

        double u, v, s;
        do
        {
            u = s_random.NextDouble() * 2 - 1;
            v = s_random.NextDouble() * 2 - 1;
            s = (u * u) + (v * v);
        }
        while (s == 0 || s >= 1);
        var z1 = u * Math.Sqrt(-2 * Math.Log(s) / s);
        _z2 = v * Math.Sqrt(-2 * Math.Log(s) / s);
        return std * z1 + mean;
    }
}
