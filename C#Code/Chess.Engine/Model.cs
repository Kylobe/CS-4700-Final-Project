using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Chess.Engine;

public sealed class Model : IDisposable
{
    public const int InPlanes = 17;
    public const int PolicyPlanes = 73;
    public const int BoardH = 8;
    public const int BoardW = 8;

    public const int InputSize = InPlanes * BoardH * BoardW;                 // 17*8*8 = 1088
    public const int PolicySize = PolicyPlanes * BoardH * BoardW;            // 73*8*8 = 4672

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string _policyName;
    private readonly string _valueName;
    public string ExecutionProvider { get; }

    public Model(
        string onnxPath,
        string inputName = "state",
        string policyOutputName = "policy_logits",
        string valueOutputName = "value",
        SessionOptions? options = null
    )
    {
        _inputName = inputName;
        _policyName = policyOutputName;
        _valueName = valueOutputName;

        if (options is null)
        {
            var (sessionOptions, provider) = CreatePreferredSessionOptions();
            ExecutionProvider = provider;
            _session = new InferenceSession(onnxPath, sessionOptions);
        }
        else
        {
            ExecutionProvider = "Custom";
            _session = new InferenceSession(onnxPath, options);
        }
    }

    private static (SessionOptions Options, string Provider) CreatePreferredSessionOptions()
    {
        SessionOptions options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };

        try
        {
            options.AppendExecutionProvider_DML(0);
            return (options, "DirectML");
        }
        catch (Exception ex)
        {
            options.Dispose();
            Program.Log($"DirectML unavailable, falling back to CPU: {ex.Message}");

            SessionOptions cpuOptions = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };

            return (cpuOptions, "CPU");
        }
    }

    /// <summary>
    /// Feed a single position encoded as 17x8x8 in NCHW flattened order (C-major):
    /// index = (c*8 + r)*8 + f.
    /// Returns policy logits (73x8x8 flattened) and value.
    /// </summary>
    public ModelOutput Invoke(float[,,] state)
    {
        if (state.GetLength(0) != InPlanes ||
            state.GetLength(1) != BoardH ||
            state.GetLength(2) != BoardW)
        {
            throw new ArgumentException(
                $"Expected shape [{InPlanes},{BoardH},{BoardW}], got " +
                $"[{state.GetLength(0)},{state.GetLength(1)},{state.GetLength(2)}]");
        }

        float[] inputArr = new float[InputSize];

        int idx = 0;
        for (int p = 0; p < InPlanes; p++)
            for (int r = 0; r < BoardH; r++)
                for (int c = 0; c < BoardW; c++)
                    inputArr[idx++] = state[p, r, c];

        var inputTensor = new DenseTensor<float>(
            inputArr,
            new[] { 1, InPlanes, BoardH, BoardW });

        var inputs = new List<NamedOnnxValue>(1)
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using var results = _session.Run(inputs);

        var policyTensor = results.First(r => r.Name == _policyName).AsTensor<float>();
        float[] policyFlat = policyTensor.ToArray();

        var valueTensor = results.First(r => r.Name == _valueName).AsTensor<float>();
        float value = valueTensor.ToArray()[0];

        if (policyFlat.Length != PolicySize)
        {
            throw new InvalidOperationException(
                $"Unexpected policy output length. Expected {PolicySize}, got {policyFlat.Length}.");
        }

        return new ModelOutput(policyFlat, value);
    }

    public void Dispose() => _session.Dispose();

    // ---- index helpers ----
    // Flat layout we use: plane-major then row then col: ((p*8 + r)*8 + c)

    public static int PolicyIndex(int plane, int row, int col)
        => ((plane * BoardH) + row) * BoardW + col;

    public static (int plane, int row, int col) DecodePolicyIndex(int idx)
    {
        int col = idx % BoardW;
        int tmp = idx / BoardW;
        int row = tmp % BoardH;
        int plane = tmp / BoardH;
        return (plane, row, col);
    }
    public ModelOutput[] InvokeBatch(float[,,,] states)
    {
        int batchSize = states.GetLength(0);

        if (states.GetLength(1) != InPlanes ||
            states.GetLength(2) != BoardH ||
            states.GetLength(3) != BoardW)
        {
            throw new ArgumentException(
                $"Expected shape [batch,{InPlanes},{BoardH},{BoardW}], got " +
                $"[{states.GetLength(0)},{states.GetLength(1)},{states.GetLength(2)},{states.GetLength(3)}]");
        }

        float[] inputArr = new float[batchSize * InputSize];

        int idx = 0;
        for (int b = 0; b < batchSize; b++)
            for (int p = 0; p < InPlanes; p++)
                for (int r = 0; r < BoardH; r++)
                    for (int c = 0; c < BoardW; c++)
                        inputArr[idx++] = states[b, p, r, c];

        var inputTensor = new DenseTensor<float>(
            inputArr,
            new[] { batchSize, InPlanes, BoardH, BoardW });

        var inputs = new List<NamedOnnxValue>(1)
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        using var results = _session.Run(inputs);

        var policyTensor = results.First(r => r.Name == _policyName).AsTensor<float>();
        float[] policyFlat = policyTensor.ToArray();

        var valueTensor = results.First(r => r.Name == _valueName).AsTensor<float>();
        float[] valueFlat = valueTensor.ToArray();

        if (policyFlat.Length != batchSize * PolicySize)
        {
            throw new InvalidOperationException(
                $"Unexpected policy output length. Expected {batchSize * PolicySize}, got {policyFlat.Length}.");
        }

        if (valueFlat.Length != batchSize)
        {
            throw new InvalidOperationException(
                $"Unexpected value output length. Expected {batchSize}, got {valueFlat.Length}.");
        }

        ModelOutput[] outputs = new ModelOutput[batchSize];

        for (int b = 0; b < batchSize; b++)
        {
            float[] onePolicy = new float[PolicySize];
            Array.Copy(policyFlat, b * PolicySize, onePolicy, 0, PolicySize);
            outputs[b] = new ModelOutput(onePolicy, valueFlat[b]);
        }

        return outputs;
    }
}
