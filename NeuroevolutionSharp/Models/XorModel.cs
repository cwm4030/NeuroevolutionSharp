using System.Diagnostics;
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using NeuroevolutionSharp.NeuralNetwork;

namespace NeuroevolutionSharp.Models;

public class XorModel : IModel<XorModel>
{
    public FullyConnectedLayer[] Layers { get; set; }

    public XorModel()
    {
        Layers = [
            new(2, 10),
            new(10, 10),
            new(10, 4),
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
        for (var i = 0; i < Layers.Length; i++)
        {
            if (i != Layers.Length - 1)
                inputs = Layers[i].FeedForward(inputs, ActivationFunctions.LeakyRelu);
            else
                inputs = Layers[i].FeedForward(inputs, ActivationFunctions.Linear);
        }
        return inputs;
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

    public void Save(string fileName)
    {
        var json = JsonSerializer.Serialize(this);
        var bytes = Encoding.UTF8.GetBytes(json);
        using var inputStream = new MemoryStream(bytes);
        using var outputStream = new MemoryStream();
        using (var gZipStream = new GZipStream(outputStream, CompressionMode.Compress, true))
        {
            inputStream.CopyTo(gZipStream);
        }
        File.WriteAllBytes(fileName, outputStream.ToArray());
    }

    public static XorModel? Open(string fileName)
    {
        var bytes = File.ReadAllBytes(fileName);
        using var inputStream = new MemoryStream(bytes);
        using var outputStream = new MemoryStream();
        using var gZipStream = new GZipStream(inputStream, CompressionMode.Decompress);
        gZipStream.CopyTo(outputStream);
        var json = Encoding.UTF8.GetString(outputStream.ToArray());
        return JsonSerializer.Deserialize<XorModel>(json);
    }

    public static void RunParameterExploringPolicyGradients()
    {
        var populationSize = 500;
        var muLearningRate = 0.2;
        var sigmaLearningRate = 0.1;
        var g = 0;
        var muOptimizer = new AdamOptimizer<XorModel>(muLearningRate).GradientAscent();
        var sigmaOptimizer = new AdamOptimizer<XorModel>(sigmaLearningRate).GradientAscent();
        var mu = Operate([], x => NormalDistribution.GetSample(0, 1));
        var sigma = Operate([], x => 1);
        var muReward = double.MinValue;

        while (g < 100000 && muReward < -0.001)
        {
            muReward = GetReward(mu);
            Console.WriteLine($"Generation {g}: {muReward}");
            g += 1;

            var epsilon = new XorModel[populationSize];
            var rewardPlus = new double[populationSize];
            var rewardNeg = new double[populationSize];
            Parallel.For(0, populationSize, i =>
            {
                epsilon[i] = Operate([sigma], x => NormalDistribution.GetSample(0, x[0]));
                var muPlus = Operate([mu, epsilon[i]], x => x[0] + x[1]);
                var muNeg = Operate([mu, epsilon[i]], x => x[0] - x[1]);
                rewardPlus[i] = GetReward(muPlus);
                rewardNeg[i] = GetReward(muNeg);
            });
            var rewardsIndex = rewardPlus.Concat(rewardNeg).Select((reward, index) => (reward, index));
            rewardsIndex = rewardsIndex.OrderBy(x => x.reward).Select((x, i) => (0.01 * (i - populationSize), x.index)).OrderBy(x => x.index);
            var rewards = rewardsIndex.Select(x => x.reward);
            rewardPlus = rewards.Take(populationSize).ToArray();
            rewardNeg = rewards.Skip(populationSize).ToArray();

            var muGradient = Operate([], x => 0);
            var sigmaGradient = Operate([], x => 0);
            Parallel.For(0, populationSize, i =>
            {
                var t = epsilon[i];
                var s = Operate([sigma, t], x => ((x[1] * x[1]) - (x[0] * x[0])) / x[0]);
                var rT = rewardPlus[i] - rewardNeg[i];
                var rS = (rewardPlus[i] + rewardNeg[i]) / 2 - muReward;
                muGradient = Operate([muGradient, t], x => x[0] + x[1] * rT);
                sigmaGradient = Operate([sigmaGradient, s], x => x[0] + x[1] * rS);
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

        if (!Debugger.IsAttached)
        {
            mu.Save("BestModel.json.zip");
            var openedModel = Open("BestModel.json.zip");
            if (openedModel != null)
            {
                prediction1 = openedModel.FeedForward([0, 0])[0];
                prediction2 = openedModel.FeedForward([1, 0])[0];
                prediction3 = openedModel.FeedForward([0, 1])[0];
                prediction4 = openedModel.FeedForward([1, 1])[0];
                Console.WriteLine();
                Console.WriteLine($"0, 0 -> {prediction1}");
                Console.WriteLine($"1, 0 -> {prediction2}");
                Console.WriteLine($"0, 1 -> {prediction3}");
                Console.WriteLine($"1, 1 -> {prediction4}");
            }
        }
    }
}
