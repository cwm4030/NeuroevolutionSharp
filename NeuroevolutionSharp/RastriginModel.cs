namespace NeuroevolutionSharp;

public class RastriginModel : IModel<RastriginModel>
{
    private readonly double[] _parameters = [0, 0, 0, 0, 0];

    public static RastriginModel GetZero()
    {
        return GetNormal(0, 0);
    }

    public static RastriginModel GetNormal(double mean, double std)
    {
        var model = new RastriginModel();
        for (var i = 0; i < model._parameters.Length; i++)
            model._parameters[i] = NormalDistribution.GetSample(mean, std);
        return model;
    }

    public static RastriginModel Operate(RastriginModel[] models, Func<double[], double> operateFunc)
    {
        var model = new RastriginModel();
        for (var i = 0; i < model._parameters.Length; i++)
            model._parameters[i] = operateFunc([.. models.Select(x => x._parameters[i])]);
        return model;
    }

    public static double GetScore(RastriginModel model)
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

    public static void RunEvolutionStrategies()
    {
        var populationSize = 500;
        var learningRate = 0.01;
        var std = 0.5;
        var g = 0;
        var optimizer = new AdamOptimizer<RastriginModel>(learningRate).GradientAscent();
        var model = GetNormal(5.12, 0);
        var score = GetScore(model);

        while (g < 100000 && score < -0.1)
        {
            score = GetScore(model);
            Console.WriteLine($"Generation {g}: {score} : {model.GetState()}");
            g += 1;

            var noiseScores = new (RastriginModel, double)[populationSize];
            for (var i = 0; i < populationSize; i++)
            {
                var noise = GetNormal(0, std);
                var newModel = Operate([model, noise], x => x[0] + x[1]);
                var newScore = GetScore(newModel);
                noiseScores[i] = (noise, newScore);
            }
            var noises = noiseScores.OrderBy(x => x.Item2).Select(x => x.Item1).ToArray();

            var gradient = GetZero();
            for (var i = 0; i < populationSize; i++)
            {
                var rankScore = 0.1 * (i - (populationSize / 2));
                gradient = Operate([gradient, noises[i]], x => x[0] + x[1] * rankScore);
            }
            gradient = Operate([gradient], x => x[0] / (std * populationSize));
            model = optimizer.Update(model, gradient);
        }
    }
}