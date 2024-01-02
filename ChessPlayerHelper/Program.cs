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
using System.Threading.Tasks;

class Program
{
    static void Main()
    {
        // //Test.TestClustering();
        // //Hey.Test();
        // // Create a 1D array
        Mat myImage = Cv2.ImRead("D:\\chesscog\\example\\myboard.jpg");

        Stopwatch stopwatch = new Stopwatch();

        stopwatch.Start();

        var ndFromImage = DetectChessBoard.FindCorners(myImage);
        
        //Mat processedImage = DetectCorners.NDArrayToMat(ndFromImage);

        stopwatch.Stop();

        TimeSpan elapsedTime = stopwatch.Elapsed;
        Console.WriteLine($"Time taken for resizing: {elapsedTime}");
        // string ipAddress = "192.168.1.77";
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
        // NDArray points = np.zeros((10,10));
        // int flag = 0;
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
        //     if(frame_count == 100)
        //     {
        //         //points =  DetectChessBoard.FindCorners(frame); 
        //         Task.Run(() =>
        //         {
        //             points = DetectChessBoard.FindCorners(frame);
        //         });

                
        //         flag = 1;
        //     }
            
        //     if(flag == 1)
        //     {
        //         var point1 = new OpenCvSharp.Point((double)points[0][0], (double)points[0][1]);
        //         var point2 = new OpenCvSharp.Point((double)points[1][0], (double)points[1][1]);
        //         var point3 = new OpenCvSharp.Point((double)points[2][0], (double)points[2][1]);
        //         var point4 = new OpenCvSharp.Point((double)points[3][0], (double)points[3][1]);
                
        //         Cv2.Line(frame, point1, point2, Scalar.Green, 6);
        //         Cv2.Line(frame, point2, point3, Scalar.Green, 6);
        //         Cv2.Line(frame, point3, point4, Scalar.Green, 6);
        //         Cv2.Line(frame, point4, point1, Scalar.Green, 6);
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