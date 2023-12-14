using System;
using NumSharp;

public static class Test
{
    public static void TestClustering()
    {
        // Example data
        var lines = np.array(new double[,] {
            {1, 0.1},
            {2, 0.2},
            {3, 0.3},
            {4, 0.4}
        });

        // Compute pairwise distances
        var distanceMatrix = PairwiseDistances(lines, AbsoluteAngleDifference);

        // Print the result
        for(int i  = 0 ; i < distanceMatrix.shape[0]; i++)
            for(int j = 0 ; j < distanceMatrix.shape[0]; j++)
            {
                Console.WriteLine("{0}",distanceMatrix[i][j]);
            }
    }

    public static NDArray PairwiseDistances(NDArray array, Func<double, double, double> distanceMetric)
    {
        int rows = array.shape[0];
        var distances = np.zeros((rows, rows));

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rows; j++)
            {
                distances[i, j] = distanceMetric(array[i, 1], array[j, 1]);
            }
        }

        return distances;
    }

    public static double AbsoluteAngleDifference(double x, double y)
    {
        double diff = MathUtils.Mod(Math.Abs(x - y), 2 * Math.PI);
        return Math.Min(diff, Math.PI - diff);
    }
}

public static class MathUtils
{
    public static double Mod(double x, double m)
    {
        return (x % m + m) % m;
    }
}
