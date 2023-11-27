/* 
    A module that combines chess board detection modules in this files
*/
using NumSharp;
using OpenCvSharp;
using System;
using System.Diagnostics;

class Program
{
    static void Main()
    {
        // Create a 1D array
        Stopwatch stopwatch = new Stopwatch();

        stopwatch.Start();

        Mat myImage = Cv2.ImRead("D:\\chesscog\\example\\myboard.jpg");
        var ndFromImage = DetectCorners.Corner(myImage);
        
        Mat processedImage = DetectCorners.NDArrayToMat(ndFromImage);

        stopwatch.Stop();

        TimeSpan elapsedTime = stopwatch.Elapsed;
        Console.WriteLine($"Time taken for resizing: {elapsedTime}");


        //Cv2.ImShow("Processed Image", processedImage);
        //Cv2.WaitKey(0);
    }
}