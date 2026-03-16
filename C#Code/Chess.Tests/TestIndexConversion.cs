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

public class IndexConversionTests
{
    [Fact]
    public void TestToAndFrom3DAnd1DIdx()
    {
        for (int idx = 0; idx < 73*8*8; idx++)
        {
            (int p, int r, int c) = ChessEnv.FlatIdxToPlaneRowCol(idx);
            int flatIdx = ChessEnv.PlaneRowColToFlatIdx(p, r, c);
            Assert.Equal(flatIdx, idx);
        }
    }
}