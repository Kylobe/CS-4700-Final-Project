using System;
using System.Diagnostics;
using System.Threading;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Factories;
using Rudzoft.ChessLib.Types;

public sealed class SearchController
{
    private readonly MCTS _mcts;
    private readonly EngineGameState _state;
    private List<Task>? _ponderTasks;
    private CancellationTokenSource? _cts;

    public SearchController(MCTS mcts, EngineGameState state)
    {
        _mcts = mcts;
        _state = state;
    }

    public void OnPositionChanged()
    {
        // Start pondering if it’s not our turn? That depends on “our color”.
        // If you want ALWAYS ponder (even on our turn while waiting), you can just start it.
        (IPosition pos, Move lastMove) = _state.SnapshotPosition();
        bool foundChild = false;
        if (!Move.Equals(Move.EmptyMove, lastMove))
        {
            foundChild = _mcts.AdvanceRoot(lastMove);
        }
        else
        {
            Program.Log("Move Was Empty!");
        }
        if (!foundChild)
        {
            _mcts.CreateRoot(pos);
            Program.Log("Recreating Root In OnPositionChanged!");
        }
        StartPondering();
    }
    public void OnPositionReset()
    {
        (IPosition pos, Move lastMove) = _state.SnapshotPosition();
        _mcts.CreateRoot(pos);
        Program.Log("Recreating Root In OnPositionChanged!");
        StartPondering();
    }
    public void StartPondering()
    {
        Stop();
        _mcts.StartBatching();

        (IPosition pos, Move lastMove) = _state.SnapshotPosition();
        int requestedWorkers = Math.Max(1, Environment.ProcessorCount - 1);
        int workerCount = Math.Min(requestedWorkers, Environment.ProcessorCount);
        workerCount = Math.Max(1, workerCount);

        _cts = new CancellationTokenSource();
        _ponderTasks = new List<Task>();

        string rootFen = pos.FenNotation;
        CancellationToken token = _cts.Token;

        for (int i = 0; i < workerCount; i++)
        {
            _ponderTasks.Add(Task.Run(() =>
            {
                IPosition localPos = GameFactory.Create(rootFen).Pos;

                while (!token.IsCancellationRequested)
                {
                    _mcts.ExpandTree(localPos);
                }
            }, token));
        }
    }

    public void Stop()
    {
        if (_cts == null)
        {
            _mcts.StopBatching();
            return;
        }

        try
        {
            _cts.Cancel();

            if (_ponderTasks is not null && _ponderTasks.Count > 0)
            {
                try
                {
                    Task.WaitAll(_ponderTasks.ToArray());
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
        finally
        {
            _cts.Dispose();
            _cts = null;
            _ponderTasks = null;
            _mcts.StopBatching();
        }
    }
    public string ThinkAndPickBestMove(int thinkMs)
    {
        Stop();
        _mcts.StartBatching();

        (IPosition pos, Move lastMove) = _state.SnapshotPosition();

        if (_mcts.root == null || _mcts.root.children.Count == 0)
        {
            _mcts.CreateRoot(pos);
        }

        int requestedWorkers = Math.Max(1, Environment.ProcessorCount - 1);
        _mcts.SearchParallel(thinkMs, requestedWorkers, pos);

        Move bestMove = _mcts.GetTopMove(pos);
        _state.ApplyEngineMove(bestMove);
        OnPositionChanged();

        return ChessEnv.GetUciFromMove(bestMove);
    }
}





