using Chess.Engine;
using System.Text;

public static class Program
{
    private static readonly object _logLock = new();
    private static readonly string _logPath =
        @"C:\CS-4800\CS-4700-Final-Project\C#Code\engine_error.log";

    public static void Log(string message)
    {
        lock (_logLock)
        {
            File.AppendAllText(
                _logPath,
                $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss.fff}] {message}{Environment.NewLine}",
                Encoding.UTF8
            );
        }
    }

    public static void Main(string[] args)
    {
        AppDomain.CurrentDomain.UnhandledException += (sender, e) =>
        {
            Log("UNHANDLED EXCEPTION:");
            Log(e.ExceptionObject?.ToString() ?? "null");
        };

        try
        {
            using var model = new Model(
                @"C:\CS-4800\CS-4700-Final-Project\C#Code\fifty_million_chess_model.onnx",
                "state",
                "policy_logits",
                "value"
            );

            var mcts = new MCTS(model, _exploreCo: 1.25f);
            var engine = new UciEngine(mcts, ourName: "TraedonEngine");

            engine.Run();
        }
        catch (Exception ex)
        {
            Log("FATAL in UCI loop:");
            Log(ex.ToString());
        }
    }
}