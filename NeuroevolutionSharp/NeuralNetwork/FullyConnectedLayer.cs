using System.Diagnostics;

namespace NeuroevolutionSharp.NeuralNetwork;

public class FullyConnectedLayer
{
    public uint NumInputs { get; set; }

    public uint NumOutputs { get; set; }

    public FullyConnectedNode[] OutputNodes { get; set; }

    public FullyConnectedLayer()
    {
        NumInputs = 0;
        NumOutputs = 0;
        OutputNodes = [];
    }

    public FullyConnectedLayer(uint numInputs, uint numOutputs)
    {
        NumInputs = numInputs;
        NumOutputs = numOutputs;
        OutputNodes = new FullyConnectedNode[numOutputs];

        for (var i = 0; i < numOutputs; i++)
            OutputNodes[i] = new FullyConnectedNode(numInputs);
    }

    public static FullyConnectedLayer Operate(FullyConnectedLayer[] layers, Func<double[], double> operateFunc)
    {
        var numInputs = layers.Length > 0 ? layers.Max(x => x.NumInputs) : 0;
        var numOutputs = layers.Length > 0 ? layers.Max(x => x.NumOutputs) : 0;
        var layer = new FullyConnectedLayer
        {
            NumInputs = numInputs,
            NumOutputs = numOutputs,
            OutputNodes = new FullyConnectedNode[numOutputs]
        };
        for (var i = 0; i < numOutputs; i++)
            layer.OutputNodes[i] = FullyConnectedNode.Operate([.. layers.Select(x => x.OutputNodes[i])], operateFunc);
        return layer;
    }

    public double[] FeedForward(double[] inputs, Func<double, double> activationFunc)
    {
        Debug.Assert(NumInputs == inputs.Length);
        var outputs = new double[NumOutputs];
        for (var i = 0; i < NumOutputs; i++)
            outputs[i] = OutputNodes[i].FeedForward(inputs, activationFunc);
        return outputs;
    }
}
