
using System;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.Enums;     // PieceTypes lives here in ChessLib
using Rudzoft.ChessLib.Extensions; // BitBoards.PopLsb is referenced from Position code paths
using Rudzoft.ChessLib.MoveGeneration;
using System.IO.Pipelines;
using Chess.Engine;
using Microsoft.ML.OnnxRuntime;
using System.Security.Principal;
using System.Net.Http.Headers;
using Rudzoft.ChessLib.Factories;

public class Node
{
    public Node? parent { get; private set; }
    public Move action { get; private set; }
    public List<Node> children { get; }
    public bool terminal { get; set; }
    public int visitCount { get; private set; }
    public float score = 0;
    public int winVal { get; set; }
    private float prior;
    public Player turn;
    public HashKey posKey;
    public Node(bool _terminal, int _winVal, float _prior, Player _turn, HashKey _posKey)
    {
        terminal = _terminal;
        winVal = _winVal;
        prior = _prior;
        turn  = _turn;
        posKey = _posKey;

        children = new List<Node>();
    }
    public void SetParent(Node newParent)
    {
        parent = newParent;
    }
    public void SetAction(Move newAction)
    {
        action = newAction;
    }
    public bool IsExpanded()
    {
        return children.Count > 0;
    }
    public static void BackPropTerminal(Node node)
    {
        bool allChildrenTerminal = true;
        bool allChildrenWinning = true;
        foreach (Node child in node.children)
        {
            if (!child.terminal)
            {
                allChildrenTerminal = false;
                allChildrenWinning = false;
                break;
            }

            else if (child.winVal < 1)
            {
                allChildrenWinning = false;
            }
        }

        if (allChildrenWinning)
        {
            node.winVal = -1;
            node.terminal = true;
            if (!(node.parent is null))
            {
                node.parent.winVal = 1;
                node.parent.terminal = true;
                if (!(node.parent.parent is null))
                {
                    BackPropTerminal(node.parent.parent);
                }
            }
        }
        else if (allChildrenTerminal)
        {
            node.winVal = 0;
            node.terminal = true;
            if (! (node.parent is null))
            {
                BackPropTerminal(node.parent);
            }
        }
    }
    /*
    public void AddChild(float prior, Move action, bool terminal, int winVal, Player sideTomMove)
    {
        Node newNode = new Node(terminal, winVal, prior, sideTomMove);
        newNode.SetAction(action);
        children.Add(newNode);
    }
    */
    public void Expand(float[] policy, IPosition pos)
    {
        bool oneDrawnChild = false;
        var legalMoves = pos.GenerateMoves();
        string originalFen = pos.FenNotation;
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
                Program.Log("FATAL in Node.Expand loop:");
                Program.Log(ex.ToString());
                Program.Log($"Tried to make the move: {move.ToString()}; in the position: {originalFen}");
            }
            bool newTerminal = false;
            int newWinVal = 0;
            if (pos.IsMate)
            {
                newTerminal = true;
                newWinVal = -1;
                terminal = true;
                winVal = 1;
                if (!(parent is null))
                {
                    BackPropTerminal(parent);
                }
            }
            else if (pos.IsDraw(pos.Ply))
            {
                newTerminal = true;
                newWinVal = 0;
                oneDrawnChild = true;
            }
            Node newNode = new Node(newTerminal, newWinVal, newPrior, pos.SideToMove, pos.State.Key);
            newNode.SetParent(this);
            newNode.SetAction(move);
            children.Add(newNode);
            pos.TakeMove(move);
            string currentFen = pos.FenNotation;
            if (currentFen != originalFen)
            {
                Program.Log("FATAL in Node.Expand loop:");
                Program.Log($"The move: {move.ToString()}, in the position: {originalFen}, resulted in: {currentFen}");
            }
        }
        if (oneDrawnChild && !terminal)
        {
            BackPropTerminal(this);
        }
    }
    public float GetUcb(float c)
    {
        float qVal = 0.5f;
        if (terminal)
        {
            if (winVal == 1)
            {
                qVal = 0f;
            }
            else if (winVal == -1)
            {
                qVal = 1f;
            }
        }
        else
        {
            if (visitCount > 0)
            {
                float ratio = score / visitCount;
                qVal = 1 - ((ratio + 1) / 2);
            }
        }
        float u = c * ((float)Math.Sqrt(visitCount + 1) / (visitCount + 1)) * prior;
        return qVal + u;
    }
    public Node Select(float c)
    {
        Node bestChild = children[0];
        float bestUcb = bestChild.GetUcb(c);
        foreach (Node child in children)
        {
            float curUcb = child.GetUcb(c);
            if (curUcb > bestUcb)
            {
                bestUcb = curUcb;
                bestChild = child;
            }
        }
        return bestChild;
    }
    public void BackPropScore(float val)
    {
        score += val;
        visitCount += 1;
        if (!(parent is null))
        {
            parent.BackPropScore(-val);
        }
    }
}
