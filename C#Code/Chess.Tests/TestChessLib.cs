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

public class ChessLibTests
{
    [Fact]
    public void TestMakeTakeMove()
    {
        IGame game = GameFactory.Create(Fen.StartPositionFen);
        game.Pos.MakeMove(Move.Create(Square.E2, Square.E4), new State());
        Assert.Equal("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2", game.Pos.FenNotation);
        game.Pos.TakeMove(Move.Create(Square.E2, Square.E4));
        IGame testGame = GameFactory.Create(Fen.StartPositionFen);
        Assert.Equal(testGame.Pos.FenNotation, game.Pos.FenNotation);
    }
    [Fact]
    public void TestEnpessant()
    {
        IGame game1 = GameFactory.Create("4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1");
        IGame game2 = GameFactory.Create("4k3/8/8/2Pp4/8/8/8/4K3 w - d6 0 1");
        Move expectedMove = Move.Create(Square.C5, Square.D6, MoveTypes.Enpassant);
        game1.Pos.MakeMove(expectedMove, new State());
        (int p, int r, int c) = (28, 4, 2);
        Move move = ChessEnv.DecodeAction((p, r, c), game2.Pos);
        game2.Pos.MakeMove(move, new State());
        Assert.Equal(game1.Pos.FenNotation, game2.Pos.FenNotation);
    }
}