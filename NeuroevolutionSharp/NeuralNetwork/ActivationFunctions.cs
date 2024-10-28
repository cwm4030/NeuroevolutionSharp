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
        var inputsAvg = inputs.Sum() / inputs.Length;
        var inputsStd = Math.Sqrt(inputs.Sum(x => (x - inputsAvg) * (x - inputsAvg)) / inputs.Length);
        inputs = inputs.Select(x => (x - inputsAvg) / inputsStd).ToArray();
        
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
