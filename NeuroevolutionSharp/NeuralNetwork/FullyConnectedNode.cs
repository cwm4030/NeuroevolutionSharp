using System.Diagnostics;

namespace NeuroevolutionSharp.NeuralNetwork;

public class FullyConnectedNode
{
    public uint NumInputs { get; set; }

    public double[] Weights { get; set; }

    public double Bias { get; set; }

    public FullyConnectedNode()
    {
        NumInputs = 0;
        Weights = [];
        Bias = 0;
    }

    public FullyConnectedNode(uint numInputs)
    {
        NumInputs = numInputs;
        Weights = new double[numInputs];
        Bias = 0;
    }

    public static FullyConnectedNode Operate(FullyConnectedNode[] nodes, Func<double[], double> operateFunc)
    {
        var numInputs = nodes.Length > 0 ? nodes.Max(x => x.NumInputs) : 0;
        var node = new FullyConnectedNode()
        {
            NumInputs = numInputs,
            Weights = new double[numInputs],
            Bias = operateFunc([.. nodes.Select(x => x.Bias)])
        };
        for (var i = 0; i < numInputs; i++)
            node.Weights[i] = operateFunc([.. nodes.Select(x => x.Weights[i])]);
        return node;
    }

    public double FeedForward(double[] inputs, Func<double, double> activationFunc)
    {
        Debug.Assert(inputs.Length == NumInputs);
        var output = Bias;
        for (var i = 0; i < NumInputs; i++)
            output += Weights[i] * inputs[i];
        return activationFunc(output);
    }
}
