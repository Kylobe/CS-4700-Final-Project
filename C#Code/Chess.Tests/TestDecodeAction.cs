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

public class ChessEnvDecodeActionTests
{
    public static IEnumerable<object[]> DecodeActionCastlingCases()
    {
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 1 2",
            15, 0, 4,
            Square.E1, Square.H1, MoveTypes.Castling
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 1 2",
            22, 0, 4,
            Square.E1, Square.A1, MoveTypes.Castling
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 1 2",
            15, 0, 3,
            Square.E8, Square.A8, MoveTypes.Castling
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 1 2",
            22, 0, 3,
            Square.E8, Square.H8, MoveTypes.Castling
        };
    }
    [Theory]
    [MemberData(nameof(DecodeActionCastlingCases))]
    public void TestDecodeCastling(string fen, int p, int r, int c, Square from, Square to, MoveTypes moveType)
    {
        IGame game = GameFactory.Create(fen);
        Move move = ChessEnv.DecodeAction((p, r, c), game.Pos);
        Move expected = Move.Create(from, to, moveType);
        Assert.True(Move.Equals(move, expected));
    }
    public static IEnumerable<object[]> DecodeActionPromoCases()
    {
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            0,6,4,
            Square.E7, Square.E8, PieceTypes.Queen
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            28,6,4,
            Square.E7, Square.F8, PieceTypes.Queen
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            35,6,4,
            Square.E7, Square.D8, PieceTypes.Queen
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            64,6,4,
            Square.E7, Square.E8, PieceTypes.Knight
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            65,6,4,
            Square.E7, Square.D8, PieceTypes.Knight
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            66,6,4,
            Square.E7, Square.F8, PieceTypes.Knight
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            67,6,4,
            Square.E7, Square.E8, PieceTypes.Bishop
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            68,6,4,
            Square.E7, Square.D8, PieceTypes.Bishop
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            69,6,4,
            Square.E7, Square.F8, PieceTypes.Bishop
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            70,6,4,
            Square.E7, Square.E8, PieceTypes.Rook
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            71,6,4,
            Square.E7, Square.D8, PieceTypes.Rook
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            72,6,4,
            Square.E7, Square.F8, PieceTypes.Rook
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            0,6,4,
            Square.D2, Square.D1, PieceTypes.Queen
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            28,6,4,
            Square.D2, Square.C1, PieceTypes.Queen
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            35,6,4,
            Square.D2, Square.E1, PieceTypes.Queen
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            64,6,4,
            Square.D2, Square.D1, PieceTypes.Knight
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            65,6,4,
            Square.D2, Square.E1, PieceTypes.Knight
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            66,6,4,
            Square.D2, Square.C1, PieceTypes.Knight
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            67,6,4,
            Square.D2, Square.D1, PieceTypes.Bishop
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            68,6,4,
            Square.D2, Square.E1, PieceTypes.Bishop
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            69,6,4,
            Square.D2, Square.C1, PieceTypes.Bishop
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            70,6,4,
            Square.D2, Square.D1, PieceTypes.Rook
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            71,6,4,
            Square.D2, Square.E1, PieceTypes.Rook
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 b - - 0 1",
            72,6,4,
            Square.D2, Square.C1, PieceTypes.Rook
        };
    }
    [Theory]
    [MemberData(nameof(DecodeActionPromoCases))]
    public void TestDecodePromotion(string fen, int p, int r, int c, Square from, Square to, PieceTypes piece)
    {
        IGame game = GameFactory.Create(fen);
        Move move = ChessEnv.DecodeAction((p, r, c), game.Pos);
        Move expected = Move.Create(from, to, MoveTypes.Promotion, piece);
        Assert.True(Move.Equals(move, expected));
    }
    public static IEnumerable<object[]> DecodeEncodeConsistencyCases()
    {
        yield return new object[]
        {
          Fen.StartPositionFen,
          Move.Create(Square.E2, Square.E4)  
        };
        yield return new object[]
        {
          "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1",
          Move.Create(Square.D7, Square.D5)  
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            Move.Create(Square.E7, Square.E8, MoveTypes.Promotion, PieceTypes.Queen)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            Move.Create(Square.E7, Square.F8, MoveTypes.Promotion, PieceTypes.Queen)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            Move.Create(Square.E7, Square.D8, MoveTypes.Promotion, PieceTypes.Queen)
        };
    }
    [Theory]
    [MemberData(nameof(DecodeEncodeConsistencyCases))]
    public void TestDecodeConsistency(string fen, Move move)
    {
        IGame game = GameFactory.Create(fen);
        (int plane, int row, int col) = ChessEnv.EncodeAction(move, game.Pos.SideToMove);
        Move decMove = ChessEnv.DecodeAction((plane, row, col), game.Pos);
        Assert.True(Move.Equals(move, decMove));
    }
}