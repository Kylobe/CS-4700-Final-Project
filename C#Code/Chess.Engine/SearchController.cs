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

    private Thread? _worker;
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
        (IPosition pos, Move lastMove) = _state.SnapshotPosition();
        _cts = new CancellationTokenSource();
        var token = _cts.Token;

        _worker = new Thread(() =>
        {
            // Ponder loop: keep expanding until canceled
            while (!token.IsCancellationRequested)
            {
                // rootGood = false here (we’re just building tree)
                // If you have “rootGood” semantics tied to main line, you can keep it false.
                _mcts.ExpandTree(pos);
            }
        })
        {
            IsBackground = true,
            Name = "PonderThread"
        };

        _worker.Start();
    }

    public void Stop()
    {
        if (_cts == null) return;

        try
        {
            _cts.Cancel();
            _worker?.Join();
        }
        catch { /* ignore */ }
        finally
        {
            _cts.Dispose();
            _cts = null;
            _worker = null;
        }
    }

    public string ThinkAndPickBestMove(int thinkMs)
    {
        Stop();
        (IPosition pos, Move lastMove) = _state.SnapshotPosition();
        HashKey posKey = pos.State.Key;
        string originalFen = pos.FenNotation;
        int earlyReturnMs = thinkMs / 5;
        //_mcts.RepairRoot(pos);

        // Ensure MCTS root matches current position
        // You can optionally call CreateRoot(pos) explicitly if needed
        // or adapt your MCTS to accept a “set root from position” method.
        var sw = Stopwatch.StartNew();
        bool stablePos = false;
        while (sw.ElapsedMilliseconds < thinkMs && !stablePos)
        {
            // IMPORTANT: ExpandTree must not leave pos mutated after returning.
            // Either ExpandTree must undo internally, or you must pass a fresh pos each call.
            for (int _ = 0; _ < 10; _++)
            {
                _mcts.ExpandTree(pos);
            }
            if (sw.ElapsedMilliseconds < earlyReturnMs)
            {
                stablePos = _mcts.PosIsStable();
            }
        }

        List<(int index, int moveCount)> moveProbs = _mcts.GetProbabilityDistribution();

        // Choose best move from probs: argmax over legal moves
        // You need a mapping from flat policy index -> Move.
        // Typically: iterate legalIndices, pick max probs[idx], decode action -> Move.
        if (pos.FenNotation != originalFen)
        {
            pos = GameFactory.Create(originalFen).Pos;
        }
        Move bestMove = _mcts.GetTopMove(pos);
        _state.ApplyEngineMove(bestMove);
        OnPositionChanged();
        return ChessEnv.GetUciFromMove(bestMove);
    }
}