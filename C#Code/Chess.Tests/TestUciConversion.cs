using System;
using System.Collections.Generic;
using Xunit;
using Rudzoft.ChessLib;
using Rudzoft.ChessLib.Types;
using Rudzoft.ChessLib.MoveGeneration;
using Rudzoft.ChessLib.Factories;
using Rudzoft.ChessLib.Fen;
using Chess.Engine;
using System.Data;
using System.Runtime.Serialization; // where your ChessEnv lives

namespace Chess.Tests;

public class UciConversionTests
{


    public static IEnumerable<object[]> UciCases()
    {
        yield return new object[]
        {
            "4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1",
            "c5d6",
            Move.Create(Square.C5, Square.D6, MoveTypes.Enpassant)
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "e1g1",
            Move.Create(Square.E1, Square.H1, MoveTypes.Castling)
        };
            yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "e1c1",
            Move.Create(Square.E1, Square.A1, MoveTypes.Castling)
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "e8g8",
            Move.Create(Square.E8, Square.H8, MoveTypes.Castling)
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "e8c8",
            Move.Create(Square.E8, Square.A8, MoveTypes.Castling)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7d8q",
            Move.Create(Square.E7, Square.D8, MoveTypes.Promotion, PieceTypes.Queen)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7e8q",
            Move.Create(Square.E7, Square.E8, MoveTypes.Promotion, PieceTypes.Queen)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7f8q",
            Move.Create(Square.E7, Square.F8, MoveTypes.Promotion, PieceTypes.Queen)
        };
            yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7d8n",
            Move.Create(Square.E7, Square.D8, MoveTypes.Promotion, PieceTypes.Knight)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7e8n",
            Move.Create(Square.E7, Square.E8, MoveTypes.Promotion, PieceTypes.Knight)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7f8n",
            Move.Create(Square.E7, Square.F8, MoveTypes.Promotion, PieceTypes.Knight)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7d8b",
            Move.Create(Square.E7, Square.D8, MoveTypes.Promotion, PieceTypes.Bishop)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7e8b",
            Move.Create(Square.E7, Square.E8, MoveTypes.Promotion, PieceTypes.Bishop)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7f8b",
            Move.Create(Square.E7, Square.F8, MoveTypes.Promotion, PieceTypes.Bishop)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7d8r",
            Move.Create(Square.E7, Square.D8, MoveTypes.Promotion, PieceTypes.Rook)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7e8r",
            Move.Create(Square.E7, Square.E8, MoveTypes.Promotion, PieceTypes.Rook)
        };
        yield return new object[]
        {
            "3b1b2/1K2P3/8/8/8/8/3p2k1/2B1B3 w - - 0 1",
            "e7f8r",
            Move.Create(Square.E7, Square.F8, MoveTypes.Promotion, PieceTypes.Rook)
        };
    }
    [Theory]
    [MemberData(nameof(UciCases))]
    public void TestUciCases(string fen, string uci, Move expectedMove)
    {
        IGame game = GameFactory.Create(fen);
        Move inputMove = ChessEnv.GetMoveFromUci(uci, game.Pos);
        Assert.Equal(inputMove, expectedMove);
        string inputUci = ChessEnv.GetUciFromMove(inputMove);
        Assert.Equal(inputUci, uci);
    }
}





