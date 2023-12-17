/* 
    A module that combines chess board detection modules in this files
*/
using NumSharp;
using OpenCvSharp;
using System;
using System.Diagnostics;
using Modules;
using Accord.Math;
using Accord.MachineLearning.Clustering;
using Aglomera;

class Program
{
    static void Main()
    {
        //Test.TestClustering();
        Hey.Test();
        // Create a 1D array
        Mat myImage = Cv2.ImRead("D:\\chesscog\\example\\myboard.jpg");

        Stopwatch stopwatch = new Stopwatch();

        stopwatch.Start();

        var ndFromImage = DetectChessBoard.FindCorners(myImage);
        
        //Mat processedImage = DetectCorners.NDArrayToMat(ndFromImage);

        stopwatch.Stop();

        TimeSpan elapsedTime = stopwatch.Elapsed;
        Console.WriteLine($"Time taken for resizing: {elapsedTime}");


        // Multiply the matrices

    }



}
