using NumSharp;
using OpenCvSharp;
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using core.OpenCvNumSharpConverter; 

namespace Modules
{
    /**
    * Static class responsible for detecting the four corners/borders of a chessboard in an image.
    * Contains methods for edge detection and resizing to assist in chessboard corner detection.
    */
    
    static class DetectChessBoard
    {    
    /**
        Find the four corner points of the chessboard for an image.
        
        Params:
            img(Mat): The image that is taken from the frame(as a MatLike)
        Returns:
             NDArray: That contains the coordinates of the four corners 
    **/
    
    public static NDArray FindCorners(Mat img)
        {
            
            NDArray npImage = MatArrayConverter.MatToNDArray(img);

            ValueTuple<float, Mat> resizedData = ResizeImage(npImage, img);
            
            // Convert from BGR to Grayscale
            Mat gray = new Mat(); 
            Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
            //Cv2.ImShow("Gray Scale", gray);
            
            //Detect Edges     
            var edges_detected = DetectEdges(gray);
            
            
            
            //Cv2.ImShow("Edge Detection", edges_detected);
            //Cv2.WaitKey(0);

            return MatArrayConverter.MatToNDArray(resizedData.Item2);
        }
        
        /**
        * Detect edges in a grayscale image using the Canny edge detection algorithm.
        * 
        * Params:
        *   grayImg (Mat): The input grayscale image (as a Mat).
        * 
        * Returns:
        *   Mat: The resulting edges after applying the Canny edge detection.
        */
        
        public static Mat DetectEdges(Mat grayImg)
        { 
            
            // Check if the data type is not UInt8
            if (grayImg.Type().Channels != 1)
            {
                // Normalize and convert to UInt8
                Cv2.Normalize(grayImg, grayImg, 0, 255, NormTypes.MinMax, MatType.CV_8U);
            }
            
            Mat Edges = new Mat();
            var LOW_THRESHOLD = 90;
            var HIGH_THRESHOLD = 400;
            var APERTURE = 3;
            
            Cv2.Canny(grayImg, Edges, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE);
            return Edges;
        }
       
        /**
        * Resize an image with a specified scale or keep it unchanged if its width is already 1200.
        * 
        * Params:
        *   npImage (NDArray): The NumSharp array representing the original image.
        *   img (Mat): The original image (as a Mat).
        * 
        * Returns:
        *   ValueTuple<float, Mat>: A tuple containing the scaling factor and the resized image.
        */
        public static ValueTuple<float, Mat> ResizeImage(NDArray npImage, Mat img)
        {        
            int height =  npImage.shape[0];
            int width  =  npImage.shape[1];
            
            if(width == 1200)  
            {
                ValueTuple<float, Mat> unProcessed = (1.00f, img);
                return unProcessed;
            }
            
            var scale = (float)((float)1200 / width);
            
            Size dimension = new Size(1200, height*scale);
            
            Cv2.Resize(img, img, dimension);
            
            ValueTuple<float, Mat> resizedImage = (scale, img);
            
            return resizedImage;
        }

    }

}