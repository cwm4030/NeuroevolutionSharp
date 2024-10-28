using System.IO.Compression;
using System.Text;
using System.Text.Json;
using NeuroevolutionSharp.Environments;
using NeuroevolutionSharp.NeuralNetwork;

namespace NeuroevolutionSharp.Models;

public class TicTacToeModel : IModel<TicTacToeModel>
{
    public FullyConnectedLayer[] Layers { get; set; }

    public TicTacToeModel()
    {
        Layers = [
            new(10, 40),
            new(40, 40),
            new(40, 20),
            new(20, 20),
            new(20, 20),
            new(20, 9)
        ];
    }

    public static TicTacToeModel Operate(TicTacToeModel[] models, Func<double[], double> operateFunc)
    {
        // preserves the shape of the model
        if (models.Length == 0)
            models = [new TicTacToeModel()];
        
        var model = new TicTacToeModel();
        for (var i = 0; i < model.Layers.Length; i++)
            model.Layers[i] = FullyConnectedLayer.Operate([.. models.Select(x => x.Layers[i])], operateFunc);
        return model;
    }

    public double[] FeedForward(double[] inputs, int[] validMoves)
    {
        for (var i = 0; i < Layers.Length; i++)
        {
            if (i != Layers.Length - 1)
                inputs = Layers[i].FeedForward(inputs, ActivationFunctions.LeakyRelu);
            else
                inputs = Layers[i].FeedForward(inputs, ActivationFunctions.Linear);
        }
        inputs = ActivationFunctions.SoftMaxFiltered(inputs, (object)validMoves);
        return inputs;
    }

    public static double GetReward(TicTacToeModel baseModel, TicTacToeModel model)
    {
        var score = 0.0;
        for (var i = 0; i < 50000; i++)
        {
            var ticTacToe = new TicTacToe();
            var result = ticTacToe.PlayOut(baseModel, model, false, true);
            if (result == TicTacToe.O || result == TicTacToe.D)
                score += 1.0;
        }

        for (var i = 0; i < 50000; i++)
        {
            var ticTacToe = new TicTacToe();
            var result = ticTacToe.PlayOut(model, baseModel, true, false);
            if (result == TicTacToe.X || result == TicTacToe.D)
                score += 1.0;
        }
        return score;
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

    public static TicTacToeModel? Open(string fileName)
    {
        try
        {
            var bytes = File.ReadAllBytes(fileName);
            using var inputStream = new MemoryStream(bytes);
            using var outputStream = new MemoryStream();
            using var gZipStream = new GZipStream(inputStream, CompressionMode.Decompress);
            gZipStream.CopyTo(outputStream);
            var json = Encoding.UTF8.GetString(outputStream.ToArray());
            return JsonSerializer.Deserialize<TicTacToeModel>(json);
        }
        catch
        {
            return null;
        }
    }

    public static void RunParameterExploringPolicyGradients()
    {
        var populationSize = 20;
        var muLearningRate = 0.2;
        var sigmaLearningRate = 0.1;
        var g = 0;
        var muOptimizer = new AdamOptimizer<TicTacToeModel>(muLearningRate).GradientAscent();
        var sigmaOptimizer = new AdamOptimizer<TicTacToeModel>(sigmaLearningRate).GradientAscent();
        var mu = Open("BestModel.json.zip") ?? Operate([], x => NormalDistribution.GetSample(0, 1));
        var sigma = Operate([], x => 1);
        var muReward = double.MinValue;
        var basePlayer = Operate([mu], x => x[0]);

        while (g < 10000)
        {
            if (g % 1000 == 0)
            {
                var ticTacToe = new TicTacToe();
                ticTacToe.DisplayPlayOut(basePlayer, mu);
            }

            muReward = GetReward(basePlayer, mu);
            Console.WriteLine($"Generation {g}: {muReward}");
            g += 1;

            if (muReward >= 75000)
            {
                basePlayer = Operate([mu], x => x[0]);
            }

            var epsilon = new TicTacToeModel[populationSize];
            var rewardPlus = new double[populationSize];
            var rewardNeg = new double[populationSize];
            Parallel.For(0, populationSize, new ParallelOptions { MaxDegreeOfParallelism = 10 }, i =>
            {
                epsilon[i] = Operate([sigma], x => NormalDistribution.GetSample(0, x[0]));
                var muPlus = Operate([mu, epsilon[i]], x => x[0] + x[1]);
                var muNeg = Operate([mu, epsilon[i]], x => x[0] - x[1]);
                rewardPlus[i] = GetReward(mu, muPlus);
                rewardNeg[i] = GetReward(mu, muNeg);
            });
            var rewardsIndex = rewardPlus.Concat(rewardNeg).Select((reward, index) => (reward, index));
            rewardsIndex = rewardsIndex.OrderBy(x => x.reward).Select((x, i) => (0.01 * (i - populationSize), x.index)).OrderBy(x => x.index);
            var rewards = rewardsIndex.Select(x => x.reward);
            rewardPlus = rewards.Take(populationSize).ToArray();
            rewardNeg = rewards.Skip(populationSize).ToArray();

            var muGradient = Operate([], x => 0);
            var sigmaGradient = Operate([], x => 0);
            Parallel.For(0, populationSize, new ParallelOptions { MaxDegreeOfParallelism = 10 }, i =>
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
            mu.Save("BestModel.json.zip");
        }
    }
}
