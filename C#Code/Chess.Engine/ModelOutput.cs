namespace Chess.Engine;

public readonly record struct ModelOutput(
    float[] PolicyLogits, // shape: [policySize] for batch=1
    float Value           // scalar for batch=1
);