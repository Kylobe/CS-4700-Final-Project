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
using System.Runtime.Serialization;
using System.Runtime.CompilerServices; // where your ChessEnv lives

namespace Chess.Tests;

public class ZobristKeyTests
{
    public static IEnumerable<object[]> ZobristTestCases()
    {
        yield return new object[]
        {
            Fen.StartPositionFen,
            Move.Create(Square.E2, Square.E4)
        };
    }
    [Theory]
    [MemberData(nameof(ZobristTestCases))]
    public static void TestZobristKey(string fen, Move move)
    {
        IGame game = GameFactory.Create(fen);
        IPosition pos = game.Pos;
        HashKey startKey = pos.State.Key;
        pos.MakeMove(move, new State());
        HashKey endKey = pos.State.Key;
        pos.TakeMove(move);
        HashKey expectedStartKey = pos.State.Key;
        pos.MakeMove(move, new State());
        HashKey expectedEndKey = pos.State.Key;
        Assert.Equal(startKey, expectedStartKey);
        Assert.Equal(endKey, expectedEndKey);
    }
}




