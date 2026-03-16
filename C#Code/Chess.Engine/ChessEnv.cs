using System;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.Enums;     // PieceTypes lives here in ChessLib
using Rudzoft.ChessLib.Extensions; // BitBoards.PopLsb is referenced from Position code paths
using Rudzoft.ChessLib.MoveGeneration;
using System.IO.Pipelines;
using System.Numerics;
using System.Text;
using System.Diagnostics;

public static class ChessEnv
{
    private const int STATE_PLANES = 17;
    private const int POLICY_PLANES = 73;

    // ---- Policy plane layout ----
    // 0..55   : queen-like moves (8 directions * 7 distances)
    // 56..63  : knight moves (8)
    // 64..72  : underpromotions (3 pieces * 3 directions)
    //
    // queen-like direction order we choose:
    // 0:N, 1:S, 2:E, 3:W, 4:NE, 5:NW, 6:SE, 7:SW
    //
    // For "to-move perspective", forward is +row.

    private static readonly (int dr, int dc)[] QueenDirs =
    {
        (+1,  0), // N
        (-1,  0), // S
        ( 0, +1), // E
        ( 0, -1), // W
        (+1, +1), // NE
        (+1, -1), // NW
        (-1, +1), // SE
        (-1, -1), // SW
    };

    // Knight deltas (dr,dc) in to-move perspective
    private static readonly (int dr, int dc)[] KnightDeltas =
    {
        (+2, +1), (+2, -1),
        (-2, +1), (-2, -1),
        (+1, +2), (+1, -2),
        (-1, +2), (-1, -2),
    };

    // Promotion directions in to-move perspective:
    // 0: forward, 1: forward-left, 2: forward-right
    private static readonly (int dr, int dc)[] PromoDeltas =
    {
        (+1,  0),
        (+1, -1),
        (+1, +1),
    };
    private static readonly Dictionary<Move, (int, int, int)>  MoveToEncodingMap = GenerateMoveEncodings();
    private static readonly Dictionary<(int, int, int), Move> EncodingToMoveMap = MoveToEncodingMap.ToDictionary(kvp => kvp.Value, kvp => kvp.Key);
    // =========================
    // Shapes:
    //   state:  (17,8,8)
    //   policy: (73,8,8)
    // =========================

    // 17-plane suggestion:
    //  0..5   : our Pawn, Knight, Bishop, Rook, Queen, King
    //  6..11  : opp Pawn, Knight, Bishop, Rook, Queen, King
    //  12..13 : our castling (K,Q)
    //  14..15 : opp castling (K,Q)
    //  16     : side-to-move (all 1s; redundant if we rotate-to-move, but keeps 17 planes)


    // Underpromotion piece order
    // 0: Knight, 1: Bishop, 2: Rook
    private static int UnderpromoPieceIndex(PieceTypes pt) => pt switch
    {
        var x when x == PieceTypes.Knight => 0,
        var x when x == PieceTypes.Bishop => 1,
        var x when x == PieceTypes.Rook   => 2,
        _ => -1
    };

    // -----------------------------
    // Public API
    // -----------------------------

    public static float[,,] EncodeBoard(IPosition pos)
    {
        var state = new float[STATE_PLANES, 8, 8];

        var turn = pos.SideToMove; // Color :contentReference[oaicite:5]{index=5}
        var us = turn;
        var them = (turn == Player.White) ? Player.Black : Player.White;

        // Piece planes: Pawn..King
        EncodePiecePlanes(state, pos, us, them, rotateToMovePerspective: true);

        // Castling planes (from perspective)
        // CanCastle(CastleRight) is provided by Position :contentReference[oaicite:6]{index=6}
        // We keep these as constant planes (all 1s if available) – common AZ style.
        bool usK = pos.CanCastle(us.IsWhite ? CastleRight.WhiteKing : CastleRight.BlackKing);
        bool usQ = pos.CanCastle(us.IsWhite ? CastleRight.WhiteQueen : CastleRight.BlackQueen);
        bool thK = pos.CanCastle(them.IsWhite ? CastleRight.WhiteKing : CastleRight.BlackKing);
        bool thQ = pos.CanCastle(them.IsWhite ? CastleRight.WhiteQueen : CastleRight.BlackQueen);

        FillPlane(state, 12, usK ? 1f : 0f);
        FillPlane(state, 13, usQ ? 1f : 0f);
        FillPlane(state, 14, thK ? 1f : 0f);
        FillPlane(state, 15, thQ ? 1f : 0f);

        // Side-to-move plane: since we rotate-to-move perspective, this is always 1.
        Square ep = pos.EnPassantSquare;

        if (ep == Square.None)
            return state;

        (int r, int c) = ToMovePerspective(ep, turn);

        state[16, r, c] = 1f;

        return state;
    }

    public static (int plane, int row, int col) EncodeAction(Move move, Player turn)
    {
        if (move.MoveType() == MoveTypes.Castling)
        {
            Square to_sq = move.ToSquare() switch
            {
              var s when s == Square.A1 => Square.C1,
              var s when s == Square.H1 => Square.G1,
              var s when s == Square.A8 => Square.C8,
              var s when s == Square.H8 => Square.G8,
              _ => throw new Exception("idk")
            };
            move = Move.Create(move.FromSquare(), to_sq);
        }
        if (move.MoveType() == MoveTypes.Enpassant)
        {
            move = Move.Create(move.FromSquare(), move.ToSquare());
        }
        if (turn == Player.Black)
        {
            move = MirrorMove(move);
        }
        if (move.PromotedPieceType() == PieceTypes.Queen)
        {
            move = Move.Create(move.FromSquare(), move.ToSquare());
        }
        return MoveToEncodingMap[move];
    }
    public static int EncodeFlatAction(Move move, Player turn)
    {
        if (turn == Player.Black)
        {
            move = MirrorMove(move);
        }
        (int p, int r, int c) = MoveToEncodingMap[move];
        return PlaneRowColToFlatIdx(p, r, c);
    }
    public static string GetUciFromMove(Move move)
    {
        string from = move.FromSquare().ToString().ToLower();
        string to   = move.ToSquare().ToString().ToLower();

        if (move.MoveType() == MoveTypes.Promotion)
        {
            char promo = move.PromotedPieceType() switch
            {
                PieceTypes.Queen  => 'q',
                PieceTypes.Rook   => 'r',
                PieceTypes.Bishop => 'b',
                PieceTypes.Knight => 'n',
                _ => throw new InvalidOperationException()
            };

            return from + to + promo;
        }
        if (move.MoveType() == MoveTypes.Castling)
        {
            string newTo = to switch
            {
                "h1" => "g1",
                "a1" => "c1",
                "h8" => "g8",
                "a8" => "c8",
                _ => to
            };
            return from + newTo;
        }

        return from + to;
    }
    public static Move DecodeAction((int plane, int row, int col) enc, IPosition pos)
    {
        Player turn = pos.SideToMove;
        Move move = EncodingToMoveMap[enc];
        Move sideMove = move;
        if (turn == Player.Black)
        {
            sideMove = MirrorMove(sideMove);
        }
        Square from_sq = sideMove.FromSquare();
        Square to_sq = sideMove.ToSquare();
        if (pos.GetPiece(from_sq).Type() == PieceTypes.Pawn && from_sq.File.AsInt() != to_sq.File.AsInt() && pos.GetPiece(to_sq) == Piece.EmptyPiece)
        {
            Move enpessantMove = Move.Create(from_sq, to_sq, MoveTypes.Enpassant);
            return enpessantMove;
        }
        if (pos.GetPiece(from_sq).Type() == PieceTypes.King && (from_sq == Square.E1 || from_sq == Square.E8))
        {
            Move castlingMove = to_sq switch
            {
                var s when s == Square.C1 => Move.Create(from_sq, Square.A1, MoveTypes.Castling),
                var s when s == Square.G1 => Move.Create(from_sq, Square.H1, MoveTypes.Castling),
                var s when s == Square.C8 => Move.Create(from_sq, Square.A8, MoveTypes.Castling),
                var s when s == Square.G8 => Move.Create(from_sq, Square.H8, MoveTypes.Castling),
                _ => Move.Create(from_sq, to_sq)
            };
            return castlingMove;
        }
        if (pos.GetPiece(from_sq).Type() == PieceTypes.Pawn && (to_sq.Rank.AsInt() == 7 || to_sq.Rank.AsInt() == 0))
        {
            Move promoMove = enc.plane switch
            {
                < 56 => Move.Create(from_sq, to_sq, MoveTypes.Promotion, PieceTypes.Queen),
                < 67 => Move.Create(from_sq, to_sq, MoveTypes.Promotion, PieceTypes.Knight),
                < 70 => Move.Create(from_sq, to_sq, MoveTypes.Promotion, PieceTypes.Bishop),
                _ => Move.Create(from_sq, to_sq, MoveTypes.Promotion, PieceTypes.Rook),
            };
            return promoMove;
        }
        return sideMove;
    }
    public static Move DecodeFlatAction(int flatIdx, IPosition pos)
    {
        (int p, int r, int c) = FlatIdxToPlaneRowCol(flatIdx);
        return DecodeAction((p, r, c), pos);
    }
    private static int GetColumn(char c)
    {
        int col = c switch
        {
            'a' => 0,
            'b' => 1,
            'c' => 2,
            'd' => 3,
            'e' => 4,
            'f' => 5,
            'g' => 6,
            'h' => 7,
            _ => throw new Exception("Character must be a-h")
        };
        return col;
    }
    public static Move GetMoveFromUci(string uci, IPosition pos)
    {

        int from_r = (int)uci[1] - '0' - 1;
        int from_c = GetColumn(uci[0]);
        int to_r = (int)uci[3]- '0' - 1;
        int to_c = GetColumn(uci[2]);
        Square from = Square.Create(from_r, from_c);
        Square to   = Square.Create(to_r, to_c);
        if (uci.Length == 5)
        {
            PieceTypes promo = uci[4] switch
            {
                'q' => PieceTypes.Queen,
                'r' => PieceTypes.Rook,
                'b' => PieceTypes.Bishop,
                'n' => PieceTypes.Knight,
                _ => throw new ArgumentException("Invalid promotion piece")
            };

            return Move.Create(from, to, MoveTypes.Promotion, promo);
        }
        if ((from == Square.E1 || from == Square.E8) && pos.GetPiece(from).Type() == PieceTypes.King)
        {
            Move kingMove = to switch
            {
                var s when s == Square.C1 => Move.Create(from, Square.A1, MoveTypes.Castling),
                var s when s == Square.G1 => Move.Create(from, Square.H1, MoveTypes.Castling),
                var s when s == Square.C8 => Move.Create(from, Square.A8, MoveTypes.Castling),
                var s when s == Square.G8 => Move.Create(from, Square.H8, MoveTypes.Castling),
                _ => Move.Create(from, to)
            };
            return kingMove;
        }
        if (from.File.AsInt() != to.File.AsInt() && pos.GetPiece(from).Type() == PieceTypes.Pawn && pos.GetPiece(to) == Piece.EmptyPiece)
        {
            return Move.Create(from, to, MoveTypes.Enpassant);
        }

        return Move.Create(from, to);
    }
    private static (int plane, int row, int col) CreateActionEncoding(Move move)
    {
        Square toSq   = move.ToSquare();
        if (move.MoveType() == MoveTypes.Castling)
        {
            if (Square.Equals(toSq, Square.A1))
            {
                toSq = Square.G1;
            }
            else if (Square.Equals(toSq, Square.H1))
            {
                toSq = Square.C1;
            }
            else if (Square.Equals(toSq, Square.A8))
            {
                toSq = Square.C8;
            }
            else if (Square.Equals(toSq, Square.H8))
            {
                toSq = Square.G8;
            }
        }
        Square fromSq = move.FromSquare();

        int fr = fromSq.Rank.AsInt();
        int fc = fromSq.File.AsInt();
        int tr = toSq.Rank.AsInt();
        int tc = toSq.File.AsInt();

        int dr = tr - fr;
        int dc = tc - fc;

        // Promotions: handle underpromotions via planes 64..72
        if (move.MoveType() == MoveTypes.Promotion && !move.IsQueenPromotion()) // :contentReference[oaicite:9]{index=9}
        {
            PieceTypes promo = move.PromotedPieceType(); // :contentReference[oaicite:10]{index=10}
            int pIdx = UnderpromoPieceIndex(promo);
            if (pIdx < 0) throw new ArgumentException($"Unsupported underpromotion piece: {promo}");

            // Must be a pawn step to last rank => dr should be +1 in our perspective
            // direction: forward (0), fwd-left(1), fwd-right(2)
            int dirIdx = (dr, dc) switch
            {
                (+1,  0) => 0,
                (+1, -1) => 1,
                (+1, +1) => 2,
                _ => throw new ArgumentException($"Invalid promotion delta dr={dr}, dc={dc}")
            };

            int plane = 64 + (pIdx * 3) + dirIdx;
            return (plane, fr, fc);
        }

        // Knight planes 56..63
        for (int k = 0; k < KnightDeltas.Length; k++)
        {
            if (KnightDeltas[k].dr == dr && KnightDeltas[k].dc == dc)
            {
                return (56 + k, fr, fc);
            }
        }

        // Queen-like planes 0..55
        int absDr = Math.Abs(dr);
        int absDc = Math.Abs(dc);

        if (absDr != absDc && absDr != 0 && absDc != 0) throw new ArgumentException($"Invalid Sliding Direction");

        int dist = Math.Max(absDr, absDc);
        if (dist < 1 || dist > 7) throw new ArgumentException($"Invalid move distance: {dist}");

        // Normalize direction
        int ndr = dr == 0 ? 0 : dr / absDr;
        int ndc = dc == 0 ? 0 : dc / absDc;

        int dir = DirIndex(ndr, ndc);
        if (dir < 0) throw new ArgumentException($"Invalid direction dr={dr}, dc={dc}");

        int planeQ = dir * 7 + (dist - 1);
        return (planeQ, fr, fc);
    }
    public static Dictionary<Move, (int, int, int)> GenerateMoveEncodings()
    {
        Dictionary<Move, (int, int, int)> EncodingMap = new Dictionary<Move, (int, int, int)>();
        for (int from_sq_int = 0; from_sq_int < 64; from_sq_int++) {
            for (int to_sq_int = 0; to_sq_int < 64; to_sq_int++)
            {
                if (from_sq_int != to_sq_int)
                {
                    Square from_sq = from_sq_int;
                    Square to_sq = to_sq_int;
                    Move move = Move.Create(from_sq, to_sq);
                    try
                    {
                        EncodingMap[move] = CreateActionEncoding(move);
                    }
                    catch
                    {
                    
                    }
                }
            }
        }
        List<PieceTypes> underPromo = [PieceTypes.Knight, PieceTypes.Bishop, PieceTypes.Rook];
        for (int from_c = 0; from_c < 8; from_c++)
        {
            for (int c_offset = -1; c_offset < 2; c_offset++)
            {
                int to_c = from_c + c_offset;
                if (to_c < 0 || to_c > 7) continue;
                foreach (PieceTypes promo in underPromo)
                {
                    int from_r = 6;
                    int to_r = 7;
                    Square from_sq = Square.Create(from_r, from_c);
                    Square to_sq = Square.Create(to_r, to_c);
                    Move move = Move.Create(from_sq, to_sq, MoveTypes.Promotion, promo);
                    try
                    {
                        EncodingMap[move] = CreateActionEncoding(move);
                    } catch
                    {
                        //do nothing
                    }
                }
            }
        }
        return EncodingMap;
    }
    public static float[] CreatePlaneActionMask(IPosition pos, out List<int> legalIndices)
    {
        legalIndices = new List<int>();
        var mask = new float[POLICY_PLANES*8*8];
        var turn = pos.SideToMove;

        // Generate legal moves (Position has move generation; README shows GenerateMoves) :contentReference[oaicite:13]{index=13}
        var moves = pos.GenerateMoves();

        foreach (var m in moves)
        {
            var (p, r, c) = EncodeAction(m, turn);
            int flatIdx = PlaneRowColToFlatIdx(p, r, c);
            mask[flatIdx] = 1f;
            legalIndices.Add(flatIdx);
        }

        return mask;
    }

    // -----------------------------
    // Internals
    // -----------------------------

    private static void EncodePiecePlanes(float[,,] state, IPosition pos, Player us, Player them, bool rotateToMovePerspective)
    {
        // PieceType order Pawn..King
        PieceTypes[] pts =
        {
            PieceTypes.Pawn, PieceTypes.Knight, PieceTypes.Bishop,
            PieceTypes.Rook, PieceTypes.Queen,  PieceTypes.King
        };

        for (int i = 0; i < pts.Length; i++)
        {
            // us pieces
            var bbUs = pos.Board.Pieces(us, pts[i]); // :contentReference[oaicite:14]{index=14}
            WriteBitboardToPlane(state, i, bbUs, us, rotateToMovePerspective);

            // them pieces
            var bbTh = pos.Board.Pieces(them, pts[i]); // :contentReference[oaicite:15]{index=15}
            WriteBitboardToPlane(state, 6 + i, bbTh, us, rotateToMovePerspective);
        }
    }

    private static void WriteBitboardToPlane(float[,,] state, int plane, BitBoard bb, Player turn, bool rotateToMovePerspective)
    {
        while (bb)
        {
            Square sq = BitBoards.PopLsb(ref bb); // :contentReference[oaicite:16]{index=16}

            (int r, int c) = rotateToMovePerspective
                ? ToMovePerspective(sq, turn)
                : (sq.Rank.AsInt(), sq.File.AsInt());

            state[plane, r, c] = 1f;
        }
    }

    private static void FillPlane(float[,,] a, int plane, float value)
    {
        for (int r = 0; r < 8; r++)
        for (int c = 0; c < 8; c++)
            a[plane, r, c] = value;
    }

    // Convert absolute Square -> (row,col) where the player-to-move is "white"
    // If turn is black, rotate 180 degrees.
    private static (int r, int c) ToMovePerspective(Square sq, Player turn)
    {
        int r = sq.Rank.AsInt(); // 0..7 for ranks 1..8 :contentReference[oaicite:17]{index=17}
        int c = sq.File.AsInt(); // 0..7 for files a..h :contentReference[oaicite:18]{index=18}

        if (turn.IsWhite) return (r, c);
        return (7 - r, 7 - c);
    }

    private static Square FromMovePerspective(int r, int c, Player turn)
    {
        if (!turn.IsWhite)
        {
            r = 7 - r;
            c = 7 - c;
        }

        // Square(rank,file) ctor expects 0..7 indices :contentReference[oaicite:19]{index=19}
        return new Square(r, c);
    }
    private static Square MirrorSquare(Square baseSquare)
    {
        int oldRow = baseSquare.Rank.AsInt();
        int oldCol = baseSquare.File.AsInt();
        int newRow = 7 - oldRow;
        int newCol = 7 - oldCol;
        return Square.Create(newRow, newCol);
    }
    private static Move MirrorMove(Move baseMove)
    {
        Square newFromSquare = MirrorSquare(baseMove.FromSquare());
        Square newToSquare = MirrorSquare(baseMove.ToSquare());
        return Move.Create(newFromSquare, newToSquare, baseMove.MoveType(), baseMove.PromotedPieceType());
    }
    private static int DirIndex(int dr, int dc)
    {
        for (int i = 0; i < QueenDirs.Length; i++)
            if (QueenDirs[i].dr == dr && QueenDirs[i].dc == dc)
                return i;
        return -1;
    }

    private static (int tr, int tc, PieceTypes? promo) DecodeQueenLike(int plane, int fr, int fc)
    {
        int dir = plane / 7;
        int dist = (plane % 7) + 1;

        var (dr, dc) = QueenDirs[dir];

        int tr = fr + dr * dist;
        int tc = fc + dc * dist;

        return (tr, tc, null);
    }

    private static (int tr, int tc, PieceTypes? promo) DecodeKnight(int plane, int fr, int fc)
    {
        int k = plane - 56;
        var (dr, dc) = KnightDeltas[k];

        return (fr + dr, fc + dc, null);
    }

    private static (int tr, int tc, PieceTypes? promo) DecodeUnderPromo(int plane, int fr, int fc)
    {
        int idx = plane - 64;     // 0..8
        int pIdx = idx / 3;       // 0..2
        int dir  = idx % 3;       // 0..2

        var (dr, dc) = PromoDeltas[dir];

        PieceTypes promo = pIdx switch
        {
            0 => PieceTypes.Knight,
            1 => PieceTypes.Bishop,
            2 => PieceTypes.Rook,
            _ => throw new ArgumentOutOfRangeException(nameof(plane))
        };

        return (fr + dr, fc + dc, promo);
    }
    public static (int p, int r, int c) FlatIdxToPlaneRowCol(int flatIdx)
    {
        int p = flatIdx / 64;
        int r = (flatIdx - p * 64) / 8;
        int c = flatIdx - p * 64 - r * 8;
        return (p, r, c);
    }
    public static int PlaneRowColToFlatIdx(int p, int r, int c)
    {
        return p * 64 + r * 8 + c;
    }

}