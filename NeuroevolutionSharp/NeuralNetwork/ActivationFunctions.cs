namespace NeuroevolutionSharp.NeuralNetwork;

public static class ActivationFunctions
{
    public static double[] LeakyRelu(double[] inputs, params object[] args)
    {
        return inputs.Select(x => x > 0 ? x : 0.1 * x).ToArray();
    }

    public static double[] Linear(double[] inputs, params object[] args)
    {
        return inputs;
    }

    public static double[] SoftMaxFiltered(double[] inputs, params object[] args)
    {
        var minInput = inputs.Min();
        var maxInput = inputs.Max();
        inputs = inputs.Select(x => (x - minInput) / (maxInput - minInput)).ToArray();

        var validIndexes = (args[0] as IEnumerable<int>)?.ToHashSet() ?? [];
        var numerators = new double[inputs.Length];
        var denominator = 0.0;
        for (var i = 0; i < inputs.Length; i++)
        {
            if (validIndexes.Contains(i))
            {
                numerators[i] = Math.Exp(inputs[i]);
                denominator += numerators[i];
            }
        }
        var outputs = numerators.Select(x => x / denominator).ToArray();
        return outputs;
    }
}
