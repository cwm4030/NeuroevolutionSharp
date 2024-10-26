namespace NeuroevolutionSharp.NeuralNetwork;

public static class ActivationFunctions
{
    public static double[] LeakyRelu(double[] inputs)
    {
        return inputs.Select(x => x > 0 ? x : 0.1 * x).ToArray();
    }

    public static double[] Linear(double[] inputs)
    {
        return inputs;
    }
}
