using System;
using System.Linq;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Factories;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.Fen;
using Rudzoft.ChessLib.MoveGeneration;
using Rudzoft.ChessLib.Protocol.UCI;

public sealed class EngineGameState
{
    private readonly object _lock = new();

    private IGame _game = GameFactory.Create(Fen.StartPositionFen);
    public List<Move> moveSequence = new List<Move>();

    public Player SideToMove
    {
        get { lock (_lock) return _game.Pos.SideToMove; }
    }

    public int Ply
    {
        get { lock (_lock) return _game.Pos.Ply; }
    }

    public void SetStartPos()
    {
        lock (_lock)
            _game = GameFactory.Create(Fen.StartPositionFen);
    }

    public void ApplyUciPosition(string positionLine)
    {
        // Supports:
        // position startpos moves ...
        // position fen <fen...> moves ...
        lock (_lock)
        {
            var parts = positionLine.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int idx = 1;

            if (parts[idx] == "startpos")
            {
                _game = GameFactory.Create(Fen.StartPositionFen);
                idx++;
            }
            else if (parts[idx] == "fen")
            {
                idx++;
                // fen is 6 tokens
                string fen = string.Join(' ', parts.Skip(idx).Take(6));
                _game = GameFactory.Create(fen);
                idx += 6;
            }

            if (idx < parts.Length && parts[idx] == "moves")
            {
                idx++;
                moveSequence.Clear();
                for (; idx < parts.Length; idx++)
                {
                    string uci = parts[idx];
                    Move m = ChessEnv.GetMoveFromUci(uci, _game.Pos);
                    moveSequence.Add(m);
                    _game.Pos.MakeMove(m, new State());
                }
            }
        }
    }

    public void ApplyEngineMove(Move m)
    {
        moveSequence.Add(m);
        _game.Pos.MakeMove(m, new State());
    }
    // Provide a safe snapshot for search:
    public (IPosition, Move) SnapshotPosition()
    {
        lock (_lock)
        {
            // Best: clone/copy from the underlying engine if available.
            // If not available, use FEN round-trip:
            string fen = _game.Pos.FenNotation;
            if (moveSequence.Count > 0)
            {
                return (GameFactory.Create(fen).Pos, moveSequence[moveSequence.Count - 1]);
            }
            else 
            {
                return (GameFactory.Create(fen).Pos, Move.EmptyMove);   
            }
        }
    }
}