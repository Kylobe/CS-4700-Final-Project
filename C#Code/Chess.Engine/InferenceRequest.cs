using System.Collections.Generic;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;

public sealed class InferenceRequest
{
    public Node Leaf { get; }
    public float[,,] State { get; }
    public List<Move> LegalMoves { get; }
    public float[] ActionMask { get; }
    public int Generation { get; }

    public InferenceRequest(Node leaf, float[,,] state, List<Move> legalMoves, float[] actionMask, int generation)
    {
        Leaf = leaf;
        State = state;
        LegalMoves = legalMoves;
        ActionMask = actionMask;
        Generation = generation;
    }
}
