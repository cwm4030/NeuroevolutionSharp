using NeuroevolutionSharp.Models;

namespace NeuroevolutionSharp.Environments;

public class TicTacToe
{
    private readonly Random _rand = new();

    public static readonly int E = 0;

    public static readonly int D = 2;

    public static readonly int X = 1;

    public static readonly int O = -1;

    public int Turn { get; private set; } = X;

    public int[] Board = new int[9];

    public int PlayOut(TicTacToeModel modelX, TicTacToeModel modelO, bool modelXTraining, bool modelOTraining)
    {
        int result;
        while (true)
        {
            result = GetBoardState();
            if (result != E) break;
            MakeMove(modelX, modelO, modelXTraining, modelOTraining);
        }
        return result;
    }

    public void DisplayPlayOut(TicTacToeModel modelX, TicTacToeModel modelO)
    {
        int result;
        while (true)
        {
            PrintBoard();
            result = GetBoardState();
            if (result != E) break;
            MakeMove(modelX, modelO, false, false);
        }
        string winner;
        if (result == X)
            winner = "X";
        else if (result == O)
            winner = "O";
        else
            winner = "Draw";
        Console.WriteLine($"Winner: {winner}");
        Console.WriteLine();
    }

    public void MakeMove(TicTacToeModel modelX, TicTacToeModel modelO, bool modelXTraining, bool modelOTraining)
    {
        var model = Turn == X ? modelX : modelO;
        var training = Turn == X ? modelXTraining : modelOTraining;
        var validMoves = GetValidMoves();
        double[] inputs = [Turn, Board[0], Board[1], Board[2], Board[3], Board[4], Board[5], Board[6], Board[7], Board[8]];
        var outputs = model.FeedForward(inputs, validMoves);

        if (training)
        {
            var moveRand = _rand.NextDouble();
            double moveSum = 0.0;
            int move = 0;
            for (var i = 0; i < outputs.Length; i++)
            {
                moveSum += outputs[i];
                if (moveSum >= moveRand)
                {
                    move = i;
                    break;
                }
            }
            Board[move] = Turn;
        }
        else
        {
            var move = outputs.Select((x, i) => (x, i)).OrderByDescending(x => x.x).First().i;
            Board[move] = Turn;
        }
        Turn = Turn == X ? O : X;
    }

    public int GetBoardState()
    {
        // Horizontal wins
        for (var i = 0; i < 3; i++)
            if (Board[i] != E && Board[i] == Board[i + 1] && Board[i + 1] == Board[i + 2])
                return Board[i];

        // Vertical wins
        for (var i = 0; i < 3; i++)
            if (Board[i] != E && Board[i] == Board[i + 3] && Board[i + 3] == Board[i + 6])
                return Board[i];

        // Diagnal wins
        if (Board[0] != E && Board[0] == Board[4] && Board[4] == Board[8])
            return Board[0];
        if (Board[2] != E && Board[2] == Board[4] && Board[4] == Board[6])
            return Board[2];

        // Draw
        for (var i = 0; i < Board.Length; i++)
        {
            if (Board[i] == E) break;
            if (i == Board.Length - 1 && Board[i] != E)
                return D;
        }

        return E;
    }

    public int[] GetValidMoves()
    {
        var validMoves = new List<int>();
        for (var i = 0; i < Board.Length; i++)
        {
            if (Board[i] == E)
                validMoves.Add(i);
        }
        return [.. validMoves];
    }

    public void PrintBoard()
    {
        var turn = Turn == X ? "X" : "O";
        Console.WriteLine($"Turn: {turn}");
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                var index = i * 3 + j;
                if (Board[index] == E)
                    Console.Write("|   ");
                else if (Board[index] == X)
                    Console.Write("| X ");
                else if (Board[index] == O)
                    Console.Write("| O ");
            }
            Console.WriteLine("|");
            Console.WriteLine("-------------");
        }
        Console.WriteLine();
    }
}
