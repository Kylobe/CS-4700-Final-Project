using System;
using System.Collections.Generic;
using Xunit;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.MoveGeneration;
using Rudzoft.ChessLib.Factories;
using Rudzoft.ChessLib.Fen;
using Chess.Engine;
using System.Data; // where your ChessEnv lives

namespace Chess.Tests;

public class ChessEnvEncodeActionTests
{

    public static IEnumerable<object[]> EncodePawnActionCases()
    {
        yield return new object[]
        {
            Fen.StartPositionFen,
            Square.E2,
            Square.E4,
            1,
            4,
            1
        };
        yield return new object[]
        {
            Fen.StartPositionFen,
            Square.E2,
            Square.E3,
            1,
            4,
            0
        };
        yield return new object[]
        {
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            Square.D7,
            Square.D5,
            1,
            4,
            1
        };
        yield return new object[]
        {
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            Square.D7,
            Square.D6,
            1,
            4,
            0
        };
    }

    [Theory]
    [MemberData(nameof(EncodePawnActionCases))]
    public void TestPawnActionEncoding(string fen, Square from, Square to, int eRow, int eCol, int ePlane)
    {
        IGame game = GameFactory.Create(fen);
        Move move = Move.Create(from, to);
        (int p, int row, int col) = ChessEnv.EncodeAction(move, game.Pos.SideToMove);
        Assert.Equal(eRow, row);
        Assert.Equal(eCol, col);
        Assert.Equal(ePlane, p);
    }

    public static IEnumerable<object[]> EncodeKnightActionCases()
    {
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.E6,
            3,
            3,
            56
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.C6,
            3,
            3,
            57
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.E2,
            3,
            3,
            58
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.C2,
            3,
            3,
            59
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.F5,
            3,
            3,
            60
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.B5,
            3,
            3,
            61
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.F3,
            3,
            3,
            62
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 w - - 0 1",
            Square.D4,
            Square.B3,
            3,
            3,
            63
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.D3,
            3,
            3,
            56
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.F3,
            3,
            3,
            57
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.D7,
            3,
            3,
            58
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.F7,
            3,
            3,
            59
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.C4,
            3,
            3,
            60
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.G4,
            3,
            3,
            61
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.C6,
            3,
            3,
            62
        };
        yield return new object[]
        {
            "3k4/8/8/4n3/3N4/8/8/4K3 b - - 0 1",
            Square.E5,
            Square.G6,
            3,
            3,
            63
        };
    }

    [Theory]
    [MemberData(nameof(EncodeKnightActionCases))]
    public void TestKnightActionEncoding(string fen, Square from, Square to, int eRow, int eCol, int ePlane)
    {
        IGame game = GameFactory.Create(fen);
        Move move = Move.Create(from, to);
        (int p, int row, int col) = ChessEnv.EncodeAction(move, game.Pos.SideToMove);
        Assert.Equal(eRow, row);
        Assert.Equal(eCol, col);
        Assert.Equal(ePlane, p);
    }

    public static IEnumerable<object[]> EncodePawnPromotionActionCases()
    {
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.E8,
            PieceTypes.Queen,
            6,
            4,
            0
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.F8,
            PieceTypes.Queen,
            6,
            4,
            28
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.D8,
            PieceTypes.Queen,
            6,
            4,
            35
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.E8,
            PieceTypes.Knight,
            6,
            4,
            64
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.D8,
            PieceTypes.Knight,
            6,
            4,
            65
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.F8,
            PieceTypes.Knight,
            6,
            4,
            66
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.E8,
            PieceTypes.Bishop,
            6,
            4,
            67
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.D8,
            PieceTypes.Bishop,
            6,
            4,
            68
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.F8,
            PieceTypes.Bishop,
            6,
            4,
            69
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.E8,
            PieceTypes.Rook,
            6,
            4,
            70
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.D8,
            PieceTypes.Rook,
            6,
            4,
            71
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 w - - 0 1",
            Square.E7,
            Square.F8,
            PieceTypes.Rook,
            6,
            4,
            72
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.D1,
            PieceTypes.Queen,
            6,
            4,
            0
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.C1,
            PieceTypes.Queen,
            6,
            4,
            28
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.E1,
            PieceTypes.Queen,
            6,
            4,
            35
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.D1,
            PieceTypes.Knight,
            6,
            4,
            64
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.E1,
            PieceTypes.Knight,
            6,
            4,
            65
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.C1,
            PieceTypes.Knight,
            6,
            4,
            66
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.D1,
            PieceTypes.Bishop,
            6,
            4,
            67
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.E1,
            PieceTypes.Bishop,
            6,
            4,
            68
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.C1,
            PieceTypes.Bishop,
            6,
            4,
            69
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.D1,
            PieceTypes.Rook,
            6,
            4,
            70
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.E1,
            PieceTypes.Rook,
            6,
            4,
            71
        };
        yield return new object[]
        {
            "3b1b1K/4P3/8/8/8/8/3p4/k1B1B3 b - - 0 1",
            Square.D2,
            Square.C1,
            PieceTypes.Rook,
            6,
            4,
            72
        };
    }
    [Theory]
    [MemberData(nameof(EncodePawnPromotionActionCases))]
    public void TestPawnPromotionActionEncoding(string fen, Square from, Square to, PieceTypes promotion, int eRow, int eCol, int ePlane)
    {
        IGame game = GameFactory.Create(fen);
        Move move = Move.Create(from, to, MoveTypes.Promotion, promotion);
        (int p, int row, int col) = ChessEnv.EncodeAction(move, game.Pos.SideToMove);
        Assert.Equal(eRow, row);
        Assert.Equal(eCol, col);
        Assert.Equal(ePlane, p);
    }

    public static IEnumerable<object[]> EncodeRookActionCases()
    {
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 w - - 0 1",
            Square.A1,
            Square.A2,
            0,
            0,
            0
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 w - - 0 1",
            Square.A1,
            Square.A8,
            0,
            0,
            6
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 w - - 0 1",
            Square.A1,
            Square.B1,
            0,
            0,
            14
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 w - - 0 1",
            Square.A1,
            Square.H1,
            0,
            0,
            20
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 b - - 0 1",
            Square.H8,
            Square.H7,
            0,
            0,
            0
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 b - - 0 1",
            Square.H8,
            Square.H1,
            0,
            0,
            6
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 b - - 0 1",
            Square.H8,
            Square.G8,
            0,
            0,
            14
        };
        yield return new object[]
        {
           "7r/6k1/8/8/8/8/1K6/R7 b - - 0 1",
            Square.H8,
            Square.A8,
            0,
            0,
            20
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 w - - 0 1",
            Square.H8,
            Square.H7,
            7,
            7,
            7
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 w - - 0 1",
            Square.H8,
            Square.H1,
            7,
            7,
            13
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 w - - 0 1",
            Square.H8,
            Square.G8,
            7,
            7,
            21
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 w - - 0 1",
            Square.H8,
            Square.A8,
            7,
            7,
            27
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 b - - 0 1",
            Square.A1,
            Square.A2,
            7,
            7,
            7
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 b - - 0 1",
            Square.A1,
            Square.A8,
            7,
            7,
            13
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 b - - 0 1",
            Square.A1,
            Square.B1,
            7,
            7,
            21
        };
        yield return new object[]
        {
            "7R/6k1/8/8/8/8/1K6/r7 b - - 0 1",
            Square.A1,
            Square.H1,
            7,
            7,
            27
        };
    }
    [Theory]
    [MemberData(nameof(EncodeRookActionCases))]
    public void TestRookActionEncoding(string fen, Square from, Square to, int eRow, int eCol, int ePlane)
    {
        IGame game = GameFactory.Create(fen);
        Move move = Move.Create(from, to);
        (int p, int row, int col) = ChessEnv.EncodeAction(move, game.Pos.SideToMove);
        Assert.Equal(eRow, row);
        Assert.Equal(eCol, col);
        Assert.Equal(ePlane, p);
    }

    public static IEnumerable<object[]> EncodeBishopActionCases()
    {
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B w - - 0 1",
            Square.A1,
            Square.B2,
            0,
            0,
            28
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B w - - 0 1",
            Square.A1,
            Square.H8,
            0,
            0,
            34
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B w - - 0 1",
            Square.H1,
            Square.G2,
            0,
            7,
            35
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B w - - 0 1",
            Square.H1,
            Square.A8,
            0,
            7,
            41
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B b - - 0 1",
            Square.H8,
            Square.G7,
            0,
            0,
            28
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B b - - 0 1",
            Square.H8,
            Square.A1,
            0,
            0,
            34
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B b - - 0 1",
            Square.A8,
            Square.B7,
            0,
            7,
            35
        };
        yield return new object[]
        {
            "b6b/2k5/8/8/8/8/5K2/B6B b - - 0 1",
            Square.A8,
            Square.H1,
            0,
            7,
            41
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b w - - 0 1",
            Square.A8,
            Square.B7,
            7,
            0,
            42
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b w - - 0 1",
            Square.A8,
            Square.H1,
            7,
            0,
            48
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b w - - 0 1",
            Square.H8,
            Square.G7,
            7,
            7,
            49
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b w - - 0 1",
            Square.H8,
            Square.A1,
            7,
            7,
            55
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b b - - 0 1",
            Square.H1,
            Square.G2,
            7,
            0,
            42
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b b - - 0 1",
            Square.H1,
            Square.A8,
            7,
            0,
            48
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b b - - 0 1",
            Square.A1,
            Square.B2,
            7,
            7,
            49
        };
        yield return new object[]
        {
            "B6B/2k5/8/8/8/8/5K2/b6b b - - 0 1",
            Square.A1,
            Square.H8,
            7,
            7,
            55
        };
        yield return new object[]
        {
            "4k3/8/3p4/4P3/8/8/8/4K3 w - - 0 1",
            Square.E5,
            Square.D6,
            4,
            4,
            35
        };
    }
    [Theory]
    [MemberData(nameof(EncodeBishopActionCases))]
    public void TestBishopActionEncoding(string fen, Square from, Square to, int eRow, int eCol, int ePlane)
    {
        IGame game = GameFactory.Create(fen);
        Move move = Move.Create(from, to);
        (int p, int row, int col) = ChessEnv.EncodeAction(move, game.Pos.SideToMove);
        Assert.Equal(eRow, row);
        Assert.Equal(eCol, col);
        Assert.Equal(ePlane, p);
    }
    [Fact]
    public void TestEverySlidingMove()
    {
        (int dr, int dc)[] queenDirs =
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

        Player[] players = { Player.White, Player.Black };

        for (int fromInt = 0; fromInt < 64; fromInt++)
        {
            Square from = fromInt;
            int fr = from.Rank.AsInt();
            int fc = from.File.AsInt();

            foreach (var (dr, dc) in queenDirs)
            {
                for (int dist = 1; ; dist++)
                {
                    int tr = fr + dr * dist;
                    int tc = fc + dc * dist;

                    if (tr < 0 || tr > 7 || tc < 0 || tc > 7)
                        break;

                    Square to = Square.Create(tr, tc);
                    Move move = Move.Create(from, to);

                    foreach (Player turn in players)
                    {
                        var (p, r, c) = ChessEnv.EncodeAction(move, turn);

                        Assert.InRange(p, 0, 72);
                        Assert.InRange(r, 0, 7);
                        Assert.InRange(c, 0, 7);
                    }
                }
            }
        }
    }
    [Fact]
    public void TestEnpessant()
    {
        Move move = Move.Create(Square.E5, Square.D6, MoveTypes.Enpassant);
        var (p, r, c) = ChessEnv.EncodeAction(move, Player.White);
        Assert.InRange(p, 0, 72);
        Assert.InRange(r, 0, 7);
        Assert.InRange(c, 0, 7);
    }
    [Fact]
    public void TestQueenPromotion()
    {
        Move move = Move.Create(Square.A7, Square.A8, MoveTypes.Promotion, PieceTypes.Queen);
        var (p, r, c) = ChessEnv.EncodeAction(move, Player.White);
        Assert.InRange(p, 0, 72);
        Assert.InRange(r, 0, 7);
        Assert.InRange(c, 0, 7);
    }
}


