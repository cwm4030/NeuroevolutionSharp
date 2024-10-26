using NeuroevolutionSharp.NeuralNetwork;

namespace NeuroevolutionSharp.Models;

public class XorModel : IModel<XorModel>
{
    public FullyConnectedLayer[] Layers { get; set; }

    public XorModel()
    {
        Layers = [
            new(2, 10),
            new(10, 4),
            new(4, 4),
            new(4, 1)
        ];
    }

    public static XorModel Operate(XorModel[] models, Func<double[], double> operateFunc)
    {
        // preserves the shape of the model
        if (models.Length == 0)
            models = [new XorModel()];
        
        var model = new XorModel();
        for (var i = 0; i < model.Layers.Length; i++)
            model.Layers[i] = FullyConnectedLayer.Operate([.. models.Select(x => x.Layers[i])], operateFunc);
        return model;
    }

    public double[] FeedForward(double[] inputs)
    {
        var outputs0 = Layers[0].FeedForward(inputs, ActivationFunctions.LeakyRelu);
        var outputs1 = Layers[1].FeedForward(outputs0, ActivationFunctions.LeakyRelu);
        var outputs2 = Layers[2].FeedForward(outputs1, ActivationFunctions.LeakyRelu);
        return Layers[3].FeedForward(outputs2, ActivationFunctions.Linear);
    }

    public static double GetReward(XorModel model)
    {
        var prediction1 = model.FeedForward([0, 0])[0];
        var prediction2 = model.FeedForward([1, 0])[0];
        var prediction3 = model.FeedForward([0, 1])[0];
        var prediction4 = model.FeedForward([1, 1])[0];

        var reward = 0.0;
        reward += (prediction1 - 0) * (prediction1 - 0);
        reward += (prediction2 - 1) * (prediction2 - 1);
        reward += (prediction3 - 1) * (prediction3 - 1);
        reward += (prediction4 - 0) * (prediction4 - 0);
        reward /= 4;
        return -1 * Math.Sqrt(reward);
    }

    public static void RunParameterExploringPolicyGradients()
    {
        var maxReward = 0;
        var populationSize = 1000;
        var muLearningRate = 0.2;
        var sigmaLearningRate = 0.1;
        var g = 0;
        var muOptimizer = new AdamOptimizer<XorModel>(muLearningRate).GradientAscent();
        var sigmaOptimizer = new AdamOptimizer<XorModel>(sigmaLearningRate).GradientAscent();
        var mu = Operate([], x => NormalDistribution.GetSample(0, 1));
        var sigma = Operate([], x => 1);
        var muReward = double.MinValue;

        while (g < 100000 && muReward < -0.01)
        {
            muReward = GetReward(mu);
            Console.WriteLine($"Generation {g}: {muReward}");
            g += 1;

            var muGradient = Operate([], x => 0);
            var sigmaGradient = Operate([], x => 0);
            Parallel.For(0, populationSize, i =>
            {
                var epsilon = Operate([sigma], x => NormalDistribution.GetSample(0, x[0]));
                var muPlus = Operate([mu, epsilon], x => x[0] + x[1]);
                var muNeg = Operate([mu, epsilon], x => x[0] - x[1]);
                var rewardPlus = GetReward(muPlus);
                var rewardNeg = GetReward(muNeg);

                var t = epsilon;
                var s = Operate([sigma, t], x => ((x[1] * x[1]) - (x[0] * x[0])) / x[0]);
                var rT = rewardPlus - rewardNeg;
                var rTNorm = 1 / (2 * maxReward - rewardPlus - rewardNeg);
                var rS = (rewardPlus + rewardNeg) / 2 - muReward;
                var rSNorm = 1 / (maxReward - muReward);
                muGradient = Operate([muGradient, t], x => x[0] + x[1] * rT * rTNorm);
                sigmaGradient = Operate([sigmaGradient, s], x => x[0] + x[1] * rS * rSNorm);
            });
            muGradient = Operate([muGradient], x => x[0] / populationSize);
            sigmaGradient = Operate([sigmaGradient], x => x[0] / populationSize);

            mu = muOptimizer.Update(mu, muGradient);
            sigma = sigmaOptimizer.Update(sigma, sigmaGradient);
        }

        var prediction1 = mu.FeedForward([0, 0])[0];
        var prediction2 = mu.FeedForward([1, 0])[0];
        var prediction3 = mu.FeedForward([0, 1])[0];
        var prediction4 = mu.FeedForward([1, 1])[0];
        Console.WriteLine();
        Console.WriteLine($"0, 0 -> {prediction1}");
        Console.WriteLine($"1, 0 -> {prediction2}");
        Console.WriteLine($"0, 1 -> {prediction3}");
        Console.WriteLine($"1, 1 -> {prediction4}");
    }
}
