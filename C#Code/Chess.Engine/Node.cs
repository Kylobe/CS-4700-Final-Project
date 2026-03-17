using System;
using System.Collections.Generic;
using System.Threading;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.Enums;
using Rudzoft.ChessLib.MoveGeneration;

public class Node
{
    private readonly object _expandLock = new object();
    private readonly object _statsLock = new object();

    private List<Node> _children = new List<Node>();

    // 0 = not expanded, 1 = expanding, 2 = expanded
    private int _expandState = 0;

    private int _visitCount = 0;
    private float _score = 0f;

    // optional worker reservation / virtual-loss bookkeeping later
    public int inFly = 0;

    public Node? parent { get; private set; }
    public Move action { get; private set; }
    public bool terminal { get; private set; }
    public int winVal { get; private set; }
    private readonly float prior;
    public Player turn { get; }

    public Node(bool terminal, int winVal, float prior, Player turn)
    {
        this.terminal = terminal;
        this.winVal = winVal;
        this.prior = prior;
        this.turn = turn;
    }

    public void SetParent(Node? newParent)
    {
        parent = newParent;
    }

    public void SetAction(Move newAction)
    {
        action = newAction;
    }

    // Safe snapshot for iteration
    public List<Node> children
    {
        get
        {
            return Volatile.Read(ref _children);
        }
    }

    public int visitCount => Volatile.Read(ref _visitCount);

    public float score
    {
        get
        {
            lock (_statsLock)
            {
                return _score;
            }
        }
    }

    public bool IsExpanded()
    {
        return Volatile.Read(ref _expandState) == 2;
    }

    public bool IsExpanding()
    {
        return Volatile.Read(ref _expandState) == 1;
    }

    public bool TryBeginExpand()
    {
        return Interlocked.CompareExchange(ref _expandState, 1, 0) == 0;
    }

    public void MarkExpanded(List<Node> newChildren)
    {
        Volatile.Write(ref _children, newChildren);
        Volatile.Write(ref _expandState, 2);
    }

    public void CancelExpand()
    {
        Volatile.Write(ref _expandState, 0);
    }
    public void IncrementVisitCount()
    {
        Interlocked.Increment(ref _visitCount);
    }

    public void SetTerminalResult(int newWinVal)
    {
        lock (_statsLock)
        {
            terminal = true;
            winVal = newWinVal;
        }
    }

    public float GetUcb(float c, int parentVisitCount)
    {
        int curVisitCount = visitCount;

        float qVal = 0.5f;
        bool isTerminal;
        int curWinVal;

        lock (_statsLock)
        {
            isTerminal = terminal;
            curWinVal = winVal;

            if (isTerminal)
            {
                if (curWinVal == 1)
                    qVal = 0f;
                else if (curWinVal == -1)
                    qVal = 1f;
            }
            else if (curVisitCount > 0)
            {
                float ratio = _score / curVisitCount;
                qVal = 1f - ((ratio + 1f) / 2f);
            }
        }

        float u = c * prior * ((float)Math.Sqrt(parentVisitCount + 1) / (curVisitCount + 1));
        return qVal + u;
    }

    public Node Select(float c, float virtualLoss)
    {
        var snapshot = children;
        if (snapshot.Count == 0)
            throw new InvalidOperationException("Cannot select from a node with no children.");

        int parentVisits = Math.Max(1, visitCount);

        Node bestChild = snapshot[0];
        float bestUcb = bestChild.GetUcb(c, parentVisits);

        for (int i = 1; i < snapshot.Count; i++)
        {
            Node child = snapshot[i];
            float curUcb = child.GetUcb(c, parentVisits);
            if (curUcb > bestUcb)
            {
                bestUcb = curUcb;
                bestChild = child;
            }
        }
        lock(bestChild._statsLock)
        {
            bestChild._score -= virtualLoss;
        }
        bestChild.IncrementVisitCount();
        return bestChild;
    }

    public void BackPropScore(float val, float virtualLoss)
    {
        Node? cur = this;
        float curVal = val;

        while (cur is not null)
        {
            lock (cur._statsLock)
            {
                cur._score += curVal + virtualLoss;
            }


            curVal = -curVal;
            cur = cur.parent;
        }
    }

    public bool TryExpand(float[] policy, IPosition pos)
    {
        if (IsExpanded())
            return false;

        if (!TryBeginExpand())
            return false;

        try
        {
            List<Node> newChildren = BuildChildren(policy, pos);
            MarkExpanded(newChildren);
            return true;
        }
        catch
        {
            CancelExpand();
            throw;
        }
    }

    private List<Node> BuildChildren(float[] policy, IPosition pos)
    {
        string originalFen = pos.FenNotation;
        List<Node> newChildren = new List<Node>();

        var legalMoves = pos.GenerateMoves();
        foreach (Move move in legalMoves)
        {
            (int p, int r, int c) = ChessEnv.EncodeAction(move, pos.SideToMove);
            int flatIdx = ChessEnv.PlaneRowColToFlatIdx(p, r, c);
            float newPrior = policy[flatIdx];

            try
            {
                pos.MakeMove(move, new State());
            }
            catch (Exception ex)
            {
                Program.Log("FATAL in Node.BuildChildren loop:");
                Program.Log(ex.ToString());
                Program.Log($"Tried to make move {move} in {originalFen}");
                throw;
            }

            bool newTerminal = false;
            int newWinVal = 0;

            if (pos.IsMate)
            {
                newTerminal = true;
                newWinVal = -1;
            }
            else if (pos.IsDraw(pos.Ply))
            {
                newTerminal = true;
                newWinVal = 0;
            }

            Node newNode = new Node(newTerminal, newWinVal, newPrior, pos.SideToMove);
            newNode.SetParent(this);
            newNode.SetAction(move);
            newChildren.Add(newNode);

            pos.TakeMove(move);

            if (pos.FenNotation != originalFen)
            {
                Program.Log("FATAL in Node.BuildChildren loop:");
                Program.Log($"Move {move} in {originalFen} resulted in corrupted position {pos.FenNotation}");
                throw new InvalidOperationException("Position corruption detected during expansion.");
            }
        }
        return newChildren;
    }
}