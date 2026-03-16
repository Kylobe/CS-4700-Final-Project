using System;
using System.Globalization;
using Rudzoft.ChessLib.Factories;
using Rudzoft.ChessLib.Types;

public sealed class UciEngine
{
    private readonly SearchController _search;
    private readonly EngineGameState _state;
    private readonly string _name;

    public UciEngine(MCTS mcts, string ourName)
    {
        _name = ourName;
        _state = new EngineGameState();
        _search = new SearchController(mcts, _state);
    }

    public void Run()
    {
        while (true)
        {
            string? line = Console.ReadLine();
            if (line == null) continue;

            line = line.Trim();
            if (line.Length == 0) continue;

            if (line == "uci")
            {
                Console.WriteLine($"id name {_name}");
                Console.WriteLine("id author Traedon Harris");
                // options would go here (setoption)
                Console.WriteLine("uciok");
            }
            else if (line == "isready")
            {
                Console.WriteLine("readyok");
            }
            else if (line == "ucinewgame")
            {
                _search.Stop();
                _state.SetStartPos();
                _search.OnPositionReset(); // restart pondering if not our turn
            }
            else if (line.StartsWith("position "))
            {
                _search.Stop();             // stop current ponder; we’re changing position
                _state.ApplyUciPosition(line);
                _search.OnPositionChanged(); // start pondering again if it’s not our turn
            }
            else if (line.StartsWith("go"))
            {
                var go = TimeManager.ParseGo(line);

                // Stop pondering; now it’s an actual search request
                _search.Stop();

                // Think for 5% of remaining time (or movetime if given)
                int thinkMs = TimeManager.ComputeThinkMs(go, _state.SideToMove);

                var best = _search.ThinkAndPickBestMove(thinkMs);

                Console.WriteLine($"bestmove {best}");
                // After bestmove, GUIs usually send a new "position ..." before next "go".
                // But you can also resume pondering immediately if you want.
            }
            else if (line == "stop")
            {
                _search.Stop();
            }
            else if (line == "quit")
            {
                _search.Stop();
                return;
            }
        }
    }
}