namespace NeuroevolutionSharp.NeuralNetwork;

public static class ActivationFunctions
{
    public static double LeakyRelu(double input)
    {
        return input > 0 ? input : 0.1 * input;
    }

    public static double Linear(double input)
    {
        return input;
    }
}
