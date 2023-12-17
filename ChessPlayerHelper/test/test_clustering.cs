using Aglomera;
using Aglomera.Linkage;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
public static class Hey
{   
class IdentityMetric<T> : IDissimilarityMetric<T>
{
    public double Calculate(T instance1, T instance2)
    {
        // Assuming T is a numeric type (double, int, etc.)
        return Convert.ToDouble(instance1.Equals(instance2) ? 0 : 1);
    }
}
 public static ISet<double> CreateInstancesFromDissimilarityMatrix(double[,] dissimilarityMatrix)
    {
        var instances = new HashSet<double>();
        for (int i = 0; i < dissimilarityMatrix.GetLength(0); i++)
        {
            for (int j = 0; j < dissimilarityMatrix.GetLength(1); j++)
            {
                instances.Add(dissimilarityMatrix[i, j]);
            }
        }
        return instances;
    }
    public static void Test()
    {
        // Example precomputed dissimilarity matrix (replace with your own data)
        var dissimilarityMatrix = new double[,]
        {
            { 0.0, 1.0, 2.0 },
            { 1.0, 0.0, 3.0 },
            { 2.0, 3.0, 0.0 },
        };
        
        // Example dissimilarity metric (identity metric since dissimilarities are precomputed)
        var identityMetric = new IdentityMetric<double>();

        // Create an instance of CompleteLinkage with the dissimilarity metric
        var completeLinkage = new AverageLinkage<double>(identityMetric);

        // Create an instance of AgglomerativeClusteringAlgorithm
        var agglomerativeClustering = new AgglomerativeClusteringAlgorithm<double>(completeLinkage);

        // Perform clustering
        var clusteringResult = agglomerativeClustering.GetClustering(CreateInstancesFromDissimilarityMatrix(dissimilarityMatrix));

        var selectedClusters = clusteringResult.Take(3);
        
        //LAST TRY
        
        //var distanceMatrix = PairwiseDistances(thetas);

        // double [][] distanceMatrixArray = new double[distanceMatrix.shape[0]][];
        
        // for(int i = 0 ; i < distanceMatrix.shape[0]; i++)
        //     distanceMatrixArray[i] = new double[distanceMatrix.shape[1]];
        
        // for(int i = 0; i < distanceMatrix.shape[0]; i++)
        //     for(int j = 0 ; j < distanceMatrix.shape[1]; j++)
        //         distanceMatrixArray[i][j] = distanceMatrix[i][j];
         


        // var metric = new PrecomputedDistanceMetric(distanceMatrix);
        
        // var linkage = new AverageLinkage<int>(metric);
        
        // var dataPoints = new HashSet<int>(Enumerable.Range(0, distanceMatrix.shape[0]));
        
        // var algorithm = new AgglomerativeClusteringAlgorithm<int>(linkage);

        // var clusteringResult = algorithm.GetClustering(dataPoints);
        

        
  
        // Access the clusters or other information from clusteringResult
        // ...

        // Example: Print the dendrogram
    }
}
