
using System.Numerics;
public class Algorithms
{
    public static float SumSIMD(float[] a)
    {
        int simdCount = Vector<float>.Count;
        int i = 0;

        Vector<float> acc = Vector<float>.Zero;

        for (; i <= a.Length - simdCount; i += simdCount)
        {
            var v = new Vector<float>(a, i);
            acc += v;
        }

        float sum = 0f;

        // Horizontal add of SIMD accumulator
        for (int j = 0; j < simdCount; j++)
            sum += acc[j];

        // Handle remainder
        for (; i < a.Length; i++)
            sum += a[i];

        return sum;
    }
    public static float[] MultiplySIMD(float[] a, float[] b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException();

        float[] result = new float[a.Length];

        int simdCount = Vector<float>.Count;
        int i = 0;

        for (; i <= a.Length - simdCount; i += simdCount)
        {
            var va = new Vector<float>(a, i);
            var vb = new Vector<float>(b, i);
            (va * vb).CopyTo(result, i);
        }

        for (; i < a.Length; i++)
            result[i] = a[i] * b[i];

        return result;
    }
    public static void MultiplyScalarInPlaceSIMD(float[] a, float scalar)
    {
        int simdCount = Vector<float>.Count;
        int i = 0;

        Vector<float> scalarVec = new Vector<float>(scalar);

        for (; i <= a.Length - simdCount; i += simdCount)
        {
            var v = new Vector<float>(a, i);
            (v * scalarVec).CopyTo(a, i);
        }

        for (; i < a.Length; i++)
            a[i] *= scalar;
    }
}


