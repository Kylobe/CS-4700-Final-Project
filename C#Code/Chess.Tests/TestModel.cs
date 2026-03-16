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

public class ModelShapeTests
{
    [Fact]
    public void PolicyHead_Output_HasShape_73x8x8()
    {
        // Path relative to test run directory.
        // Easiest: set "Copy to Output Directory" on chess_model.onnx,
        // or point this at a known absolute path.
        const string onnxPath = "C:\\CS-4800\\C-sharp-chess-bot\\chess_model.onnx";

        using var model = new Model(
            onnxPath,
            inputName: "state",
            policyOutputName: "policy_logits",
            valueOutputName: "value"
        );

        IGame game = GameFactory.Create(Fen.StartPositionFen);

        float[,,] state = ChessEnv.EncodeBoard(game.Pos);

        ModelOutput output = model.Invoke(state);

        // Policy logits should be (73,8,8) => 4672
        Assert.NotNull(output.PolicyLogits);
        Assert.Equal(Model.PolicySize, output.PolicyLogits.Length);
        Assert.Equal(73 * 8 * 8, output.PolicyLogits.Length);
    }
    [Fact]
    public void TestValueHeadOutput()
    {
        const string onnxPath = "C:\\CS-4800\\C-sharp-chess-bot\\chess_model.onnx";

        using var model = new Model(
            onnxPath,
            inputName: "state",
            policyOutputName: "policy_logits",
            valueOutputName: "value"
        );
        IGame game = GameFactory.Create("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1");
        float[,,] state = ChessEnv.EncodeBoard(game.Pos);
        ModelOutput output = model.Invoke(state);
        float valueOutput = output.Value;
        Console.Write(valueOutput);
        Assert.True(valueOutput > 0);
    }
}