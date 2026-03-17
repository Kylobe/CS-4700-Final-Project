using System;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.Enums;     // PieceTypes lives here in ChessLib
using Rudzoft.ChessLib.Extensions; // BitBoards.PopLsb is referenced from Position code paths
using Rudzoft.ChessLib.MoveGeneration;
using System.IO.Pipelines;
using Chess.Engine;
using Microsoft.ML.OnnxRuntime;
using Rudzoft.ChessLib.Factories;
using Rudzoft.ChessLib.Fen;
using System.Runtime.CompilerServices;

public class MCTS
{
    private Model model;
    private float exploreCo;
    public Node? root { get; private set; }
    public MCTS(Model _model, float _exploreCo)
    {
        if (_model is null)
        {
            throw new NullReferenceException("Model Can't Be Null");
        }
        model = _model;
        exploreCo = _exploreCo;
    }
    public void CreateRoot(IPosition pos)
    {
        bool terminal = false;
        int winVal = 0;

        if (pos.IsMate)
        {
            terminal = true;
            winVal = -1;
        }
        else if (pos.IsDraw(pos.Ply))
        {
            terminal = true;
        }

        root = new Node(terminal, winVal, 0f, pos.SideToMove);

        if (!terminal)
        {
            List<int> legalIndices;
            float[] mask = ChessEnv.CreatePlaneActionMask(pos, out legalIndices);
            float[,,] state = ChessEnv.EncodeBoard(pos);
            ModelOutput output = model.Invoke(state);
            float[] policy = output.PolicyLogits;
            float[] masked = Algorithms.MultiplySIMD(policy, mask);
            float sum = Algorithms.SumSIMD(masked);

            if (sum > 0f)
            {
                float inv = 1f / sum;
                Algorithms.MultiplyScalarInPlaceSIMD(masked, inv);
            }

            root.TryExpand(masked, pos);
        }

        Program.Log("Recreating Root!");
    }
    public bool AdvanceRoot(Move action)
    {
        if (root == null)
            throw new NullReferenceException("Can't advance root when null.");

        foreach (var child in root.children)
        {
            if (child.action.Equals(action))
            {
                root = child;
                root.SetParent(null);
                return true;
            }
        }

        Program.Log($"Move: {action} not found among root.");
        return false;
    }
    public void ExpandTree(IPosition pos)
    {
        float virtualLoss = 1f;
        if (root is null)
        {
            CreateRoot(pos);
        }

        Node curNode = root!;
        List<Move> movesFromRoot = new List<Move>();

        while (curNode.IsExpanded())
        {
            var snapshot = curNode.children;
            if (snapshot.Count == 0)
                break;

            curNode = curNode.Select(exploreCo, virtualLoss);
            pos.MakeMove(curNode.action, new State());
            movesFromRoot.Add(curNode.action);
        }

        float value;
        bool terminated = curNode.terminal;

        if (terminated)
        {
            value = curNode.winVal;
        }
        else
        {
            List<int> legalIndices;
            float[] mask = ChessEnv.CreatePlaneActionMask(pos, out legalIndices);
            float[,,] state = ChessEnv.EncodeBoard(pos);
            ModelOutput output = model.Invoke(state);

            float[] policy = output.PolicyLogits;
            value = output.Value;

            float[] masked = Algorithms.MultiplySIMD(policy, mask);
            float sum = Algorithms.SumSIMD(masked);

            if (sum > 0f)
            {
                float inv = 1f / sum;
                Algorithms.MultiplyScalarInPlaceSIMD(masked, inv);
            }

            curNode.TryExpand(masked, pos);
        }

        curNode.BackPropScore(value, virtualLoss);

        for (int i = movesFromRoot.Count - 1; i >= 0; i--)
        {
            pos.TakeMove(movesFromRoot[i]);
        }
    }
    public bool PosIsStable()
    {
        if (root is null || root.children.Count == 0)
        {
            return false;
        }
        if (root.terminal)
        {
            return true;
        }
        int totalVisitCount = 0;
        int topVisitCount = 0;
        foreach (Node child in root.children)
        {
            int curVisitCount = child.visitCount;
            totalVisitCount += curVisitCount;
            if (curVisitCount > topVisitCount)
            {
                topVisitCount = curVisitCount;
            }
        }
        if (totalVisitCount == 0)
        {
            return false;
        }
        return ((double)topVisitCount / totalVisitCount) > 0.8d;
    }

    public Move GetTopMove(IPosition pos)
    {
        //RepairRoot(pos);
        if (root.children.Count > 0)
        {
            Node bestChild = root.children[0];
            foreach (Node child in root.children)
            {
                if (child.visitCount > bestChild.visitCount)
                {
                    bestChild = child;
                }
            }
            return bestChild.action;
        }
        Program.Log("Root Node Not Expanded!");
        var legalMoves = pos.GenerateMoves();
        foreach (Move move in legalMoves)
        {
            return move;
        }
        return Move.Create(Square.A1, Square.A1);
    }
    public List<(int, int)> GetProbabilityDistribution()
    {
        if (root is null)
        {
            throw new NullReferenceException("Can't Get Probability With Null Root");
        }
        List<(int idx, int prob)> visitCounts = new List<(int idx, int prob)>();
        if (root.terminal && root.winVal == 1)
        {
            foreach (Node child in root.children)
            {
                if (child.terminal && child.winVal == -1)
                {
                    (int p, int r, int c) = ChessEnv.EncodeAction(child.action, root.turn);
                    int flatIdx = ChessEnv.PlaneRowColToFlatIdx(p, r, c);
                    visitCounts.Add((flatIdx, child.visitCount));
                    return visitCounts;
                }
            }
        }
        foreach (Node child in root.children)
        {
            (int p, int r, int c) = ChessEnv.EncodeAction(child.action, root.turn);
            int flatIdx = ChessEnv.PlaneRowColToFlatIdx(p, r, c);
            visitCounts.Add((flatIdx, child.visitCount));
        }
        return visitCounts;
    }
    public void SearchParallel(int thinkMs, int requestedWorkers, IPosition rootPos)
    {
        if (root is null)
        {
            CreateRoot(rootPos);
        }

        int workerCount = Math.Min(requestedWorkers, Environment.ProcessorCount);
        workerCount = Math.Max(1, workerCount);

        using CancellationTokenSource cts = new CancellationTokenSource();
        cts.CancelAfter(thinkMs);

        List<Task> tasks = new List<Task>();

        string rootFen = rootPos.FenNotation;

        for (int i = 0; i < workerCount; i++)
        {
            tasks.Add(Task.Run(() =>
            {
                IPosition localPos = GameFactory.Create(rootFen).Pos;

                while (!cts.Token.IsCancellationRequested)
                {
                    ExpandTree(localPos);
                }
            }, cts.Token));
        }

        try
        {
            Task.WaitAll(tasks.ToArray());
        }
        catch (AggregateException ex)
        {
            foreach (Exception inner in ex.InnerExceptions)
            {
                if (inner is not OperationCanceledException)
                {
                    Program.Log(inner.ToString());
                }
            }
        }
    }
}
