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
using System.Diagnostics;
using System.Collections.Concurrent;
using System.Collections.Generic;

public class MCTS
{
    private Model model;
    private float exploreCo;
    public Node? root { get; private set; }

    private float virtualLoss = 1;
    private int _rootGeneration = 0;
    private long _nodesSearched = 0;
    private long _lastReportedNodes = 0;
    private readonly Stopwatch _npsWatch = Stopwatch.StartNew();
    private readonly ConcurrentQueue<InferenceRequest> _inferenceQueue = new();
    private readonly AutoResetEvent _hasInferenceWork = new(false);
    private readonly ManualResetEventSlim _queueCanAcceptWork = new(true);

    private Task? _batchTask;
    private CancellationTokenSource? _batchCts;

    private readonly int _maxBatchSize = 16;
    private readonly int _maxBatchDelayMs = 0;
    private readonly int _queueHighWaterMultiplier = 8;
    private readonly int _queueLowWaterMultiplier = 4;
    private long _expandCalls = 0;
    private long _selectedEdges = 0;
    private long _terminalHits = 0;
    private long _expandingSkips = 0;
    private long _tryBeginExpandFails = 0;
    private long _requestsEnqueued = 0;
    private long _staleRequestsDropped = 0;
    private long _batchCount = 0;
    private long _batchLeafCount = 0;
    private long _batchWaitTicks = 0;
    private long _batchPackTicks = 0;
    private long _modelTicks = 0;
    private long _maskTicks = 0;
    private long _expandFinishTicks = 0;
    private long _backpropTicks = 0;
    private long _encodeBoardTicks = 0;
    private long _generateMovesTicks = 0;
    private long _queueBackpressureWaitTicks = 0;
    private long _queueBackpressurePauses = 0;

    public long TotalNodesSearched => Interlocked.Read(ref _nodesSearched);
    public int CurrentGeneration => Volatile.Read(ref _rootGeneration);

    public MCTS(Model _model, float _exploreCo)
    {
        if (_model is null)
        {
            throw new NullReferenceException("Model Can't Be Null");
        }
        model = _model;
        exploreCo = _exploreCo;
    }
    public void StopBatching()
    {
        _queueCanAcceptWork.Set();

        if (_batchCts is not null)
        {
            _batchCts.Cancel();

            try
            {
                _batchTask?.Wait();
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

            _batchTask = null;
            _batchCts.Dispose();
            _batchCts = null;
            _hasInferenceWork.Set();
        }

        while (_inferenceQueue.TryDequeue(out var req))
        {
            req.Leaf.CancelExpand();
            Interlocked.Decrement(ref req.Leaf.inFly);
        }

        UpdateQueueBackpressureState(0);
    }
    public void StartBatching()
    {
        StopBatching();
        _batchCts = new CancellationTokenSource();
        CancellationToken token = _batchCts.Token;
        _batchTask = Task.Run(() => BatchLoop(token), token);
    }
    private void BatchLoop(CancellationToken token)
    {
        while (!token.IsCancellationRequested)
        {
            long waitStart = Stopwatch.GetTimestamp();
            _hasInferenceWork.WaitOne(1);
            Interlocked.Add(ref _batchWaitTicks, Stopwatch.GetTimestamp() - waitStart);

            List<InferenceRequest> batch = new();
            Stopwatch sw = Stopwatch.StartNew();

            while (batch.Count < _maxBatchSize)
            {
                while (batch.Count < _maxBatchSize && _inferenceQueue.TryDequeue(out var req))
                {
                    batch.Add(req);
                }

                UpdateQueueBackpressureState(_inferenceQueue.Count);

                if (batch.Count > 0 && sw.ElapsedMilliseconds >= _maxBatchDelayMs)
                    break;

                if (batch.Count == 0)
                {
                    if (token.IsCancellationRequested) return;
                    _hasInferenceWork.WaitOne(1);
                }
                else
                {
                    Thread.Yield();
                }
            }

            if (batch.Count == 0)
                continue;

            RunInferenceBatch(batch);
            UpdateQueueBackpressureState(_inferenceQueue.Count);
        }
    }
    private void RunInferenceBatch(List<InferenceRequest> batch)
    {
        Interlocked.Increment(ref _batchCount);
        Interlocked.Add(ref _batchLeafCount, batch.Count);

        int batchSize = batch.Count;
        float[,,,] states = new float[batchSize, 17, 8, 8];

        long packStart = Stopwatch.GetTimestamp();
        for (int b = 0; b < batchSize; b++)
        {
            float[,,] state = batch[b].State;
            for (int p = 0; p < 17; p++)
            for (int r = 0; r < 8; r++)
            for (int c = 0; c < 8; c++)
                states[b, p, r, c] = state[p, r, c];
        }
        Interlocked.Add(ref _batchPackTicks, Stopwatch.GetTimestamp() - packStart);

        long modelStart = Stopwatch.GetTimestamp();
        ModelOutput[] outputs = model.InvokeBatch(states);
        Interlocked.Add(ref _modelTicks, Stopwatch.GetTimestamp() - modelStart);

        for (int i = 0; i < batch.Count; i++)
        {
            var req = batch[i];
            if (req.Generation != CurrentGeneration)
            {
                Interlocked.Increment(ref _staleRequestsDropped);
                req.Leaf.CancelExpand();
                Interlocked.Decrement(ref req.Leaf.inFly);
                continue;
            }

            var output = outputs[i];
            var leaf = req.Leaf;

            try
            {
                if (leaf.terminal)
                {
                    long terminalBackpropStart = Stopwatch.GetTimestamp();
                    leaf.BackPropScore(leaf.winVal, virtualLoss);
                    Interlocked.Add(ref _backpropTicks, Stopwatch.GetTimestamp() - terminalBackpropStart);
                    Interlocked.Increment(ref _nodesSearched);
                    continue;
                }

                long maskStart = Stopwatch.GetTimestamp();
                float[] masked = Algorithms.MultiplySIMD(output.PolicyLogits, req.ActionMask);
                float sum = Algorithms.SumSIMD(masked);

                if (sum > 0f)
                {
                    float inv = 1f / sum;
                    Algorithms.MultiplyScalarInPlaceSIMD(masked, inv);
                }
                Interlocked.Add(ref _maskTicks, Stopwatch.GetTimestamp() - maskStart);

                long finishExpandStart = Stopwatch.GetTimestamp();
                leaf.FinishExpand(masked, req.LegalMoves);
                Interlocked.Add(ref _expandFinishTicks, Stopwatch.GetTimestamp() - finishExpandStart);
                long inferenceBackpropStart = Stopwatch.GetTimestamp();
                leaf.BackPropScore(output.Value, 0f);
                Interlocked.Add(ref _backpropTicks, Stopwatch.GetTimestamp() - inferenceBackpropStart);
                Interlocked.Increment(ref _nodesSearched);
            }
            catch
            {
                leaf.CancelExpand();
                throw;
            }
            finally
            {
                Interlocked.Decrement(ref leaf.inFly);
            }
        }
    }
    public void CreateRoot(IPosition pos)
    {
        Interlocked.Increment(ref _rootGeneration);
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
            List<Move> legalMoves = new();
            foreach (Move move in pos.GenerateMoves())
            {
                legalMoves.Add(move);
            }

            float[] mask = ChessEnv.CreatePlaneActionMask(pos.SideToMove, legalMoves);
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

            root.TryExpand(masked, legalMoves);
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
        Interlocked.Increment(ref _expandCalls);

        if (root is null)
            CreateRoot(pos);

        Node curNode = root!;
        List<Move> movesFromRoot = new();
        List<Node> selectedPath = new();

        try
        {

            while (curNode.IsExpanded())
            {
                var snapshot = curNode.children;
                if (snapshot.Count == 0)
                    break;

                Node? nextNode = curNode.Select(exploreCo, virtualLoss);
                if (nextNode is null)
                {
                    RollBackSelectionPath(selectedPath);
                    return;
                }

                curNode = nextNode;
                selectedPath.Add(curNode);
                Interlocked.Increment(ref _selectedEdges);
                pos.MakeMove(curNode.action, new State());
                movesFromRoot.Add(curNode.action);
            }


            WaitForQueueCapacity();

            if (curNode.terminal)
            {
                Interlocked.Increment(ref _terminalHits);
                long terminalNodeBackpropStart = Stopwatch.GetTimestamp();
                curNode.BackPropScore(curNode.winVal, virtualLoss);
                Interlocked.Add(ref _backpropTicks, Stopwatch.GetTimestamp() - terminalNodeBackpropStart);
                return;
            }

            if (pos.IsMate)
            {
                curNode.SetTerminalResult(-1);
                Interlocked.Increment(ref _terminalHits);
                long mateBackpropStart = Stopwatch.GetTimestamp();
                curNode.BackPropScore(-1, virtualLoss);
                Interlocked.Add(ref _backpropTicks, Stopwatch.GetTimestamp() - mateBackpropStart);
                return;
            }

            if (pos.IsDraw(pos.Ply))
            {
                curNode.SetTerminalResult(0);
                Interlocked.Increment(ref _terminalHits);
                long drawBackpropStart = Stopwatch.GetTimestamp();
                curNode.BackPropScore(0, virtualLoss);
                Interlocked.Add(ref _backpropTicks, Stopwatch.GetTimestamp() - drawBackpropStart);
                return;
            }

            if (curNode.IsExpanding())
            {
                Interlocked.Increment(ref _expandingSkips);
                RollBackSelectionPath(selectedPath);
                return;
            }

            if (!curNode.TryBeginExpand())
            {
                Interlocked.Increment(ref _tryBeginExpandFails);
                RollBackSelectionPath(selectedPath);
                return;
            }

            Interlocked.Increment(ref curNode.inFly);

            long encodeStart = Stopwatch.GetTimestamp();
            float[,,] state = ChessEnv.EncodeBoard(pos);
            Interlocked.Add(ref _encodeBoardTicks, Stopwatch.GetTimestamp() - encodeStart);
            long movesStart = Stopwatch.GetTimestamp();
            List<Move> legalMoves = new();
            foreach (Move move in pos.GenerateMoves())
            {
                legalMoves.Add(move);
            }

            float[] actionMask = ChessEnv.CreatePlaneActionMask(pos.SideToMove, legalMoves);
            Interlocked.Add(ref _generateMovesTicks, Stopwatch.GetTimestamp() - movesStart);
            var req = new InferenceRequest(curNode, state, legalMoves, actionMask, CurrentGeneration);
            _inferenceQueue.Enqueue(req);
            Interlocked.Increment(ref _requestsEnqueued);
            UpdateQueueBackpressureState(_inferenceQueue.Count);
            _hasInferenceWork.Set();
        }
        finally
        {
            for (int i = movesFromRoot.Count - 1; i >= 0; i--)
                pos.TakeMove(movesFromRoot[i]);
        }
        Interlocked.Increment(ref _nodesSearched);
    }
    private void RollBackSelectionPath(List<Node> selectedPath)
    {
        for (int i = selectedPath.Count - 1; i >= 0; i--)
        {
            Node node = selectedPath[i];
            node.UndoVirtualLoss(virtualLoss);
            node.DecrementVisitCount();
        }
    }
    public bool PosIsStable(float[] prevVisitCounts, float[] curVisitCounts, int maxThinkMs, int elapsedTime, float threshHold = 0.995f)
    {
        if (prevVisitCounts.Length != curVisitCounts.Length)
        {
            throw new Exception("Array Sizes Differ");
        }
        int remainingTime = maxThinkMs - elapsedTime;
        if (remainingTime <= 0)
        {
            return true;
        }
        float dot = 0;
        float prevMag = 0;
        float curMag = 0;
        for (int i = 0; i < prevVisitCounts.Length; i++)
        {
            dot += prevVisitCounts[i] * curVisitCounts[i];
            prevMag += (float)Math.Pow(prevVisitCounts[i], 2);
            curMag += (float)Math.Pow(curVisitCounts[i], 2);
        }
        prevMag = (float)Math.Sqrt(prevMag);
        curMag = (float)Math.Sqrt(curMag);
        float similarity = dot / (prevMag * curMag);
        float timeRatio = (float)remainingTime / maxThinkMs;
        return similarity >= threshHold;
    }

    public Move GetTopMove(IPosition pos)
    {
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
    public float[] GetCurVisitCounts()
    {
        if (root is null)
        {
            return Array.Empty<float>();
        }
        lock(root._statsLock)
        {
            float[] visitCounts = new float[root.children.Count];
            int i = 0;
            foreach(Node child in root.children)
            {
                visitCounts[i] = child.visitCount;
                i++;
            }
            return visitCounts;
        }
    }

    public double GetNodesPerSecond()
    {
        double seconds = _npsWatch.Elapsed.TotalSeconds;
        if (seconds <= 0.0) return 0.0;
        return Interlocked.Read(ref _nodesSearched) / seconds;
    }

    public double GetIntervalNodesPerSecond()
    {
        long current = Interlocked.Read(ref _nodesSearched);
        long previous = Interlocked.Exchange(ref _lastReportedNodes, current);
        double seconds = _npsWatch.Elapsed.TotalSeconds;
        _npsWatch.Restart();

        if (seconds <= 0.0) return 0.0;
        return (current - previous) / seconds;
    }
    public void SearchParallel(int thinkMs, int requestedWorkers, IPosition rootPos)
    {
        if (root is null)
        {
            CreateRoot(rootPos);
        }

        ResetProfiling();
        _npsWatch.Restart();

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

        Program.Log($"Final NPS: {GetNodesPerSecond():F0}, Total nodes: {TotalNodesSearched}");
        Program.Log(GetProfilingSummary());
    }

    private void ResetProfiling()
    {
        Interlocked.Exchange(ref _nodesSearched, 0);
        Interlocked.Exchange(ref _lastReportedNodes, 0);
        Interlocked.Exchange(ref _expandCalls, 0);
        Interlocked.Exchange(ref _selectedEdges, 0);
        Interlocked.Exchange(ref _terminalHits, 0);
        Interlocked.Exchange(ref _expandingSkips, 0);
        Interlocked.Exchange(ref _tryBeginExpandFails, 0);
        Interlocked.Exchange(ref _requestsEnqueued, 0);
        Interlocked.Exchange(ref _staleRequestsDropped, 0);
        Interlocked.Exchange(ref _batchCount, 0);
        Interlocked.Exchange(ref _batchLeafCount, 0);
        Interlocked.Exchange(ref _batchWaitTicks, 0);
        Interlocked.Exchange(ref _batchPackTicks, 0);
        Interlocked.Exchange(ref _modelTicks, 0);
        Interlocked.Exchange(ref _maskTicks, 0);
        Interlocked.Exchange(ref _expandFinishTicks, 0);
        Interlocked.Exchange(ref _backpropTicks, 0);
        Interlocked.Exchange(ref _encodeBoardTicks, 0);
        Interlocked.Exchange(ref _generateMovesTicks, 0);
        Interlocked.Exchange(ref _queueBackpressureWaitTicks, 0);
        Interlocked.Exchange(ref _queueBackpressurePauses, 0);
        UpdateQueueBackpressureState(_inferenceQueue.Count);
    }

    private string GetProfilingSummary()
    {
        static double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;

        long expandCalls = Interlocked.Read(ref _expandCalls);
        long selectedEdges = Interlocked.Read(ref _selectedEdges);
        long terminalHits = Interlocked.Read(ref _terminalHits);
        long expandingSkips = Interlocked.Read(ref _expandingSkips);
        long tryBeginExpandFails = Interlocked.Read(ref _tryBeginExpandFails);
        long requestsEnqueued = Interlocked.Read(ref _requestsEnqueued);
        long staleDropped = Interlocked.Read(ref _staleRequestsDropped);
        long batchCount = Interlocked.Read(ref _batchCount);
        long batchLeafCount = Interlocked.Read(ref _batchLeafCount);

        return
            $"PROFILE " +
            $"expand_calls={expandCalls}, " +
            $"selected_edges={selectedEdges}, " +
            $"terminal_hits={terminalHits}, " +
            $"expanding_skips={expandingSkips}, " +
            $"try_begin_expand_fails={tryBeginExpandFails}, " +
            $"requests_enqueued={requestsEnqueued}, " +
            $"stale_dropped={staleDropped}, " +
            $"batches={batchCount}, " +
            $"avg_batch_size={(batchCount > 0 ? (double)batchLeafCount / batchCount : 0):F2}, " +
            $"batch_wait_ms={Ms(Interlocked.Read(ref _batchWaitTicks)):F1}, " +
            $"batch_pack_ms={Ms(Interlocked.Read(ref _batchPackTicks)):F1}, " +
            $"model_ms={Ms(Interlocked.Read(ref _modelTicks)):F1}, " +
            $"mask_ms={Ms(Interlocked.Read(ref _maskTicks)):F1}, " +
            $"finish_expand_ms={Ms(Interlocked.Read(ref _expandFinishTicks)):F1}, " +
            $"backprop_ms={Ms(Interlocked.Read(ref _backpropTicks)):F1}, " +
            $"encode_board_ms={Ms(Interlocked.Read(ref _encodeBoardTicks)):F1}, " +
            $"generate_moves_ms={Ms(Interlocked.Read(ref _generateMovesTicks)):F1}, " +
            $"queue_backpressure_wait_ms={Ms(Interlocked.Read(ref _queueBackpressureWaitTicks)):F1}, " +
            $"queue_backpressure_pauses={Interlocked.Read(ref _queueBackpressurePauses)}";
    }

    private void WaitForQueueCapacity()
    {
        if (_queueCanAcceptWork.IsSet)
            return;

        long waitStart = Stopwatch.GetTimestamp();
        _queueCanAcceptWork.Wait();
        Interlocked.Add(ref _queueBackpressureWaitTicks, Stopwatch.GetTimestamp() - waitStart);
    }

    private void UpdateQueueBackpressureState(int queueCount)
    {
        int highWaterMark = _maxBatchSize * _queueHighWaterMultiplier;
        int lowWaterMark = _maxBatchSize * _queueLowWaterMultiplier;

        if (queueCount >= highWaterMark)
        {
            if (_queueCanAcceptWork.IsSet)
            {
                _queueCanAcceptWork.Reset();
                Interlocked.Increment(ref _queueBackpressurePauses);
            }
        }
        else if (queueCount <= lowWaterMark)
        {
            _queueCanAcceptWork.Set();
        }
    }
}
