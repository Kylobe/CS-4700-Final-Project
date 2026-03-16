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

public class ChessEnvEncodeBoardTests
{

    // A few positions that stress different move types:
    // - start position (lots of normals + knights)
    // - position with castling available
    // - promotions and underpromotions
    // - en passant possibility (optional depending on your encode_action coverage)
    public static IEnumerable<object[]> Fens()
    {
        yield return new object[] { Fen.StartPositionFen };

        // Both sides can castle (simple):
        yield return new object[] { "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1" };

        // Underpromotion opportunities (white to move, pawn on 7th rank with capture options)
        // White pawn on g7 can promote forward to g8 and capture to f8/h8 depending on pieces.
        yield return new object[] { "5n1r/6P1/8/8/8/8/8/4K3 w - - 0 1" };

        // Black underpromotion opportunity (black to move, pawn on b2 going to b1)
        yield return new object[] { "4k3/8/8/8/8/8/1p6/4K3 b - - 0 1" };
    }

    [Fact]
    public void TestStartingPosition()
    {
        IGame game = GameFactory.Create(Fen.StartPositionFen);
        float[,,] encoding = ChessEnv.EncodeBoard(game.Pos);
        //Pawns
        for (int i = 0; i < 8; i++) 
        {
            Assert.Equal(1f, encoding[0, 1, i]);
            Assert.Equal(1f, encoding[6, 6, i]);
        }
        //Knights
        Assert.Equal(1f, encoding[1, 0, 1]);
        Assert.Equal(1f, encoding[1, 0, 6]);
        Assert.Equal(1f, encoding[7, 7, 1]);
        Assert.Equal(1f, encoding[7, 7, 6]);
        //Bishops
        Assert.Equal(1f, encoding[2, 0, 2]);
        Assert.Equal(1f, encoding[2, 0, 5]);
        Assert.Equal(1f, encoding[8, 7, 2]);
        Assert.Equal(1f, encoding[8, 7, 5]);
        //Rooks
        Assert.Equal(1f, encoding[3, 0, 0]);
        Assert.Equal(1f, encoding[3, 0, 7]);
        Assert.Equal(1f, encoding[9, 7, 0]);
        Assert.Equal(1f, encoding[9, 7, 7]);
        //Queens
        Assert.Equal(1f, encoding[4, 0, 3]);
        Assert.Equal(1f, encoding[10, 7, 3]);
        //Kings
        Assert.Equal(1f, encoding[5, 0, 4]);
        Assert.Equal(1f, encoding[11, 7, 4]);
        //Castling Rights
        for (int plane = 12; plane < 16; plane++)
        {
            for (int row = 0; row < 8; row++)
            {
                for (int col = 0; col < 8; col++)
                {
                    Assert.Equal(1f, encoding[plane, row, col]);
                }
            }
        }
        //Enpessant
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                Assert.Equal(0f, encoding[16, row, col]);
            }
        }
    }

    [Fact]
    public void TestStartingPositionForBlack()
    {
        var game = GameFactory.Create("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
        float[,,] encoding = ChessEnv.EncodeBoard(game.Pos);
        //Pawns
        for (int i = 0; i < 8; i++) 
        {
            Assert.Equal(1f, encoding[0, 1, i]);
            Assert.Equal(1f, encoding[6, 6, i]);
        }
        //Knights
        Assert.Equal(1f, encoding[1, 0, 1]);
        Assert.Equal(1f, encoding[1, 0, 6]);
        Assert.Equal(1f, encoding[7, 7, 1]);
        Assert.Equal(1f, encoding[7, 7, 6]);
        //Bishops
        Assert.Equal(1f, encoding[2, 0, 2]);
        Assert.Equal(1f, encoding[2, 0, 5]);
        Assert.Equal(1f, encoding[8, 7, 2]);
        Assert.Equal(1f, encoding[8, 7, 5]);
        //Rooks
        Assert.Equal(1f, encoding[3, 0, 0]);
        Assert.Equal(1f, encoding[3, 0, 7]);
        Assert.Equal(1f, encoding[9, 7, 0]);
        Assert.Equal(1f, encoding[9, 7, 7]);
        //Queens
        Assert.Equal(1f, encoding[4, 0, 4]);
        Assert.Equal(1f, encoding[10, 7, 4]);
        //Kings
        Assert.Equal(1f, encoding[5, 0, 3]);
        Assert.Equal(1f, encoding[11, 7, 3]);
        //Castling Rights
        for (int plane = 12; plane < 16; plane++)
        {
            for (int row = 0; row < 8; row++)
            {
                for (int col = 0; col < 8; col++)
                {
                    Assert.Equal(1f, encoding[plane, row, col]);
                }
            }
        }
        //Enpessant
        for (int row = 0; row < 8; row++)
        {
            for (int col = 0; col < 8; col++)
            {
                Assert.Equal(0f, encoding[16, row, col]);
            }
        }
    }
    [Fact]
    public void TestEnpessant()
    {
        var game = GameFactory.Create("4k3/8/8/1Pp5/8/8/8/4K3 w - c6 0 1");
        float[,,] encoding = ChessEnv.EncodeBoard(game.Pos);
        //Enpessant
        Assert.Equal(1f, encoding[16, 5, 2]);
    }
}