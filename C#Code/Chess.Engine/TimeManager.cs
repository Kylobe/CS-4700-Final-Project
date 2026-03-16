using System;

public readonly struct GoParams
{
    public readonly int? MoveTimeMs;
    public readonly int? WTimeMs;
    public readonly int? BTimeMs;
    public readonly int? WIncMs;
    public readonly int? BIncMs;

    public GoParams(int? movetime, int? wtime, int? btime, int? winc, int? binc)
    {
        MoveTimeMs = movetime;
        WTimeMs = wtime;
        BTimeMs = btime;
        WIncMs = winc;
        BIncMs = binc;
    }
}

public static class TimeManager
{
    public static GoParams ParseGo(string line)
    {
        // Very small parser: go wtime <ms> btime <ms> winc <ms> binc <ms> movetime <ms>
        int? movetime = null, wtime = null, btime = null, winc = null, binc = null;

        var parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        for (int i = 1; i < parts.Length - 1; i++)
        {
            string k = parts[i];
            if (!int.TryParse(parts[i + 1], out int v)) continue;

            switch (k)
            {
                case "movetime": movetime = v; break;
                case "wtime": wtime = v; break;
                case "btime": btime = v; break;
                case "winc": winc = v; break;
                case "binc": binc = v; break;
            }
        }

        return new GoParams(movetime, wtime, btime, winc, binc);
    }

    public static int ComputeThinkMs(GoParams go, Rudzoft.ChessLib.Types.Player sideToMove)
    {
        const int minThinkMs = 10;       // don’t do 0ms searches
        const int maxThinkMs = 30_000;   // clamp so you don’t burn forever accidentally

        if (go.MoveTimeMs.HasValue)
            return Math.Clamp((int)(go.MoveTimeMs.Value * 0.05f), minThinkMs, maxThinkMs);

        int remaining = sideToMove == Rudzoft.ChessLib.Types.Player.White
            ? (go.WTimeMs ?? 0)
            : (go.BTimeMs ?? 0);

        if (remaining <= 0) return minThinkMs;

        int think = (int)(remaining * 0.05f);

        // Optional: include a little increment
        int inc = sideToMove == Rudzoft.ChessLib.Types.Player.White
            ? (go.WIncMs ?? 0)
            : (go.BIncMs ?? 0);

        think += (int)(inc * 0.05f);

        return Math.Clamp(think, minThinkMs, maxThinkMs);
    }
}