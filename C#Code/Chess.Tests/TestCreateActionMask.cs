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

public class ChessEnvCreateActionMaskTests
{
    public static IEnumerable<object[]> CastlingMaskCases()
    {
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            new (int p, int r, int c)[] 
            {
                (15, 0, 4),
                (22, 0, 4)
            }
        };
        yield return new object[]
        {
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            new (int p, int r, int c)[] 
            {
                (15, 0, 3),
                (22, 0, 3)
            }
        };
    }
    [Theory]
    [MemberData(nameof(CastlingMaskCases))]
    public void TestCastlingRights(string fen, (int p, int r, int c)[] ones)
    {
        IGame game = GameFactory.Create(fen);
        List<int> legalIndices;
        float[] mask = ChessEnv.CreatePlaneActionMask(game.Pos, out legalIndices);
        foreach (var (p, r, c) in ones)
        {
            int flatIdx = ChessEnv.PlaneRowColToFlatIdx(p, r, c);
            Assert.Equal(1f, mask[flatIdx]);
        }
    }
}