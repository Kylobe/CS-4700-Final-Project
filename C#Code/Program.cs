public static class Program
{
    public static void Main(string[] args)
    {
        using var model = new Model("chess_model.onnx", "state", "policy_logits", "value");
        var mcts = new MCTS(model, _exploreCo: 1.25f);

        var engine = new UciEngine(mcts, ourName: "TraedonEngine");
        engine.Run();
    }
}