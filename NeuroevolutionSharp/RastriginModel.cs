namespace NeuroevolutionSharp;

public class RastriginModel : IModel<RastriginModel>
{
    private readonly double[] _parameters = new double[10];

    public static RastriginModel Operate(RastriginModel[] models, Func<double[], double> operateFunc)
    {
        var model = new RastriginModel();
        for (var i = 0; i < model._parameters.Length; i++)
            model._parameters[i] = operateFunc([.. models.Select(x => x._parameters[i])]);
        return model;
    }

    public static double GetReward(RastriginModel model)
    {
        var a = 10.0;
        var score = a * model._parameters.Length;
        for (var i = 0; i < model._parameters.Length; i++)
            score += (model._parameters[i] * model._parameters[i]) - (a * Math.Cos(2 * Math.PI * model._parameters[i]));
        return -1 * score;
    }

    public string GetState()
    {
        return string.Join(", ", _parameters);
    }

    public static void RunParameterExploringPolicyGradients()
    {
        var populationSize = 1000;
        var muLearningRate = 0.2;
        var sigmaLearningRate = 0.1;
        var g = 0;
        var muOptimizer = new AdamOptimizer<RastriginModel>(muLearningRate).GradientAscent();
        var sigmaOptimizer = new AdamOptimizer<RastriginModel>(sigmaLearningRate).GradientAscent();
        var mu = Operate([], x => 5.12);
        var sigma = Operate([], x => 2);
        double muReward = double.MinValue;

        while (g < 100000 && muReward < -0.01)
        {
            muReward = GetReward(mu);
            Console.WriteLine($"Generation {g}: {muReward}");
            g += 1;

            var rewardPlus = new double[populationSize];
            var rewardNeg = new double[populationSize];
            var epsilon = new RastriginModel[populationSize];
            for (var i = 0; i < populationSize; i++)
            {
                var noise = Operate([sigma], x => NormalDistribution.GetSample(0, x[0]));
                var muPlus = Operate([mu, noise], x => x[0] + x[1]);
                var muNeg = Operate([mu, noise], x => x[0] - x[1]);
                epsilon[i] = noise;
                rewardPlus[i] = GetReward(muPlus);
                rewardNeg[i] = GetReward(muNeg);
            }
            var rewards = rewardPlus.Concat(rewardNeg).ToArray();
            var rewardsAvg = rewards.Sum() / rewards.Length;
            var rewardsStd = Math.Sqrt(rewards.Sum(x => (x - rewardsAvg) * (x - rewardsAvg)) / rewards.Length);
            rewards = rewards.Select(x => (x - rewardsAvg) / rewardsStd).ToArray();
            rewardPlus = rewards.Take(populationSize).ToArray();
            rewardNeg = rewards.Skip(populationSize).ToArray();

            var muGradient = Operate([], x => 0);
            var sigmaGradient = Operate([], x => 0);
            for (var i = 0; i < populationSize; i++)
            {
                var t = epsilon[i];
                var s = Operate([sigma, t], x => ((x[1] * x[1]) - (x[0] * x[0])) / x[0]);
                var rT = rewardPlus[i] - rewardNeg[i];
                var sT = (rewardPlus[i] + rewardNeg[i]) / 2 - muReward;
                muGradient = Operate([muGradient, t], x => x[0] + rT * x[1]);
                sigmaGradient = Operate([sigmaGradient, s], x => x[0] + sT * x[1]);
            }
            muGradient = Operate([muGradient], x => x[0] / populationSize);
            sigmaGradient = Operate([sigmaGradient], x => x[0] / populationSize);

            mu = muOptimizer.Update(mu, muGradient);
            sigma = sigmaOptimizer.Update(sigma, sigmaGradient);
        }
        Console.WriteLine();
        Console.WriteLine(mu.GetState());
    }
}
