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
public class MCTSTests
{
    [Fact]
    public void TestPositionUnchanged()
    {
        const string onnxPath = "C:\\CS-4800\\C-sharp-chess-bot\\chess_model.onnx";

        using var model = new Model(
            onnxPath,
            inputName: "state",
            policyOutputName: "policy_logits",
            valueOutputName: "value"
        );

        MCTS mcts = new MCTS(model, 2f);
        IGame game = GameFactory.Create(Fen.StartPositionFen);
        mcts.CreateRoot(game.Pos);
        for (int i = 0; i < 100; i++)
        {
            mcts.ExpandTree(game.Pos);
        }
        Assert.Equal(Fen.StartPositionFen, game.Pos.FenNotation);
    }
}

