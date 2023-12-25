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
        //Hey.Test();
        // Create a 1D array
        Mat myImage = Cv2.ImRead("D:\\chesscog\\example\\myboard.jpg");

        Stopwatch stopwatch = new Stopwatch();

        stopwatch.Start();

        var ndFromImage = DetectChessBoard.FindCorners(myImage);
        
        //Mat processedImage = DetectCorners.NDArrayToMat(ndFromImage);

        stopwatch.Stop();

        TimeSpan elapsedTime = stopwatch.Elapsed;
        Console.WriteLine($"Time taken for resizing: {elapsedTime}");
        // string ipAddress = "192.168.1.40";
        // int port = 8080;

        // // Create a VideoCapture object with the IP address and port
        // VideoCapture capture = new VideoCapture($"http://{ipAddress}:{port}/video");

        // // Check if the VideoCapture object is opened successfully
        // if (!capture.IsOpened())
        // {
        //     Console.WriteLine("Error opening video stream from the mobile device.");
        //     return;
        // }
        // int frame_count = 0;
        // // Create a window for displaying the live video
        // Cv2.NamedWindow("Live Video", WindowFlags.Normal);
        // Cv2.ResizeWindow("Live Video", 1200, 900);
        // int frameWidth = 1200;
        // int frameHeight = 900;

        // // Set the frame size properties
        // capture.Set(VideoCaptureProperties.FrameWidth, frameWidth);
        // capture.Set(VideoCaptureProperties.FrameHeight, frameHeight);

        // while (true)
        // {
        //     // Read a frame from the video stream
        //     Mat frame = new Mat();
        //     capture.Read(frame);
        //     frame_count++;
        //     // Check if the frame is empty
        //     if (frame.Empty())
        //     {
        //         Console.WriteLine("Error reading frame from the video stream.");
        //         break;
        //     }
        //     if(frame_count == 200)
        //     {
        //         var all_intersection_points =  DetectChessBoard.FindCorners(frame); 


        //     }

        //     // Display the frame in the "Live Video" window
        //     Cv2.ImShow("Live Video", frame);

        //     Console.WriteLine("Frame Count {0}",frame_count);
        //     // Break the loop if the user presses the 'Esc' key
        //     if (Cv2.WaitKey(1) == 27)
        //         break;
        // }

        // // Release the VideoCapture object and close the window when done
        // capture.Release();
        // Cv2.DestroyAllWindows();

        // Multiply the matrices

    }



}
  // Mat imgWithPoints = img.Clone();

        // for(int i = 0 ; i < all_intersection_points.shape[0]; i++) 
        // {
        //     for(int j = 0; j < all_intersection_points.shape[1]; j++)
        //     {
        //         double x = all_intersection_points[i][j][0];
        //         double y = all_intersection_points[i][j][1];

        //         Cv2.Circle(imgWithPoints, new OpenCvSharp.Point(x, y), 5, new Scalar(0, 255, 0), -1);
        //     }
        // }
        // Cv2.ImShow("Image with Points", imgWithPoints);
        // Cv2.WaitKey(0);