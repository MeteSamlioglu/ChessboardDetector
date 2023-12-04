using NumSharp;
using OpenCvSharp;
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using core.OpenCvNumSharpConverter; 
using Data;
using System.Linq;

namespace Modules
{
    
    /**
    * Static class responsible for detecting the four corners/borders of a chessboard in an image.
    * Contains methods for edge detection and resizing to assist in chessboard corner detection.
    */
    
static class DetectChessBoard
{    
    
    public static NDArray FindCorners(Mat img)
    {
    /**
        Find the four corner points of the chessboard for an image.
        
        Params:
            img(Mat): The image that is taken from the frame(as a MatLike)
        Returns:
            NDArray: That contains the coordinates of the four corners 
    **/

        ValueTuple<float, Mat> resizedData = ResizeImage(img);

        // Convert from BGR to Grayscale
        Mat gray = new Mat(); 

        Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
        
        var edges_detected = DetectEdges(gray);
        
        NDArray np = MatArrayConverter.MatToNDArray(edges_detected);
    
        var detected_lines = DetectLines(edges_detected);


        if (detected_lines.shape[0] > 400 )
        {
           Console.WriteLine("Too many lines are detected in the image!");
           return 0;
        }


        return MatArrayConverter.MatToNDArray(resizedData.Item2);
    }
    
    public static Mat DetectEdges(Mat grayImg)
    { 
        /**
        * Detect edges in a grayscale image using the Canny edge detection algorithm.
        * 
        * Params:
        *   grayImg (Mat): The input grayscale image (as a Mat).
        * 
        * Returns:
        *   Mat: The resulting edges after applying the Canny edge detection.
        */        
        if (grayImg.Type().Channels != 1)
        {
            // Normalize and convert to UInt8
            Cv2.Normalize(grayImg, grayImg, 0, 255, NormTypes.MinMax, MatType.CV_8U);
        }
        
        Mat Edges = new Mat();
        
        Cv2.Canny(grayImg,  Edges, 
                    CONFIGURATION.EDGE_DETECTION.LOW_THRESHOLD, 
                    CONFIGURATION.EDGE_DETECTION.HIGH_THRESHOLD, 
                    CONFIGURATION.EDGE_DETECTION.APERTURE);
        
        
        return Edges;
    }
        static float DegreesToRadians(float degrees)
        {
            double DegreesToRadiansMultiplier = np.pi / 180.0;

            return (float)(degrees * DegreesToRadiansMultiplier);
        }

        public static NDArray DetectLines(Mat Edges)
        {

            var lines = Cv2.HoughLines(Edges,1, np.pi/360, CONFIGURATION.LINE_DETECTION.THRESHOLD);
              
            if(lines == null ||lines.Length == 0)
            {
                Console.WriteLine("Chessboard is not detected");
                return np.zeros(0);
            }    
            
            var linesSqueezed = MatArrayConverter.MatToNDArray(lines);
            
            
            var hesse_form = fixNegativeRhoInHesseForm(linesSqueezed);
            
     
            if (CONFIGURATION.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION)
            {
                
                var treshold = DegreesToRadians(CONFIGURATION.LINE_DETECTION.DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES);
                
                Console.WriteLine("Treshold  {0}",treshold);
                
                var vmask = np.abs(hesse_form[$":", "1"]) < treshold;
                var hmask = np.abs(hesse_form[$":", "1"] - (np.pi / 2)) < treshold;
                var mask = np.logical_or(vmask, hmask);
           
                var _lines = hesse_form[mask];
                return _lines;
            }
            
            return np.zeros(0);
        }

        public static NDArray fixNegativeRhoInHesseForm(NDArray lines)
        {
            
            var linesCopy = lines.Clone() as NDArray;

            // Check if lines is null
            if (lines == null)
            {
                // Handle the case where lines is null
                return np.zeros(0);
            }

            // Assuming lines has a shape of (..., 2) since you are using lines[..., 0] and lines[..., 1]
            int rows = linesCopy.shape[0];
            int cols = linesCopy.shape[1];

            // Assuming rho is the last column and theta is the second last column
            int rhoIndex = cols - 1;
            int thetaIndex = cols - 2;

            // Iterate through the array and apply the fix
            for (int i = 0; i < rows; i++)
            {
                float rhoValue = linesCopy[i][0];
                
                if (rhoValue < 0)
                {
                    linesCopy[i][0] = - rhoValue;
                    linesCopy[i][1] = linesCopy[i][1] - np.pi;
                }
            }

            return linesCopy;
        }
        
        public static ValueTuple<float, Mat> ResizeImage(Mat img)
        {        
            /**
            * Resize an image with a specified scale or keep it unchanged if its width is already 1200.
            * 
            * Params:
            *   img (Mat): The original image (as a Mat).
            * 
            * Returns:
            *   ValueTuple<float, Mat>: A tuple containing the scaling factor and the resized image.
            */
            Size sizeOfImg = img.Size();
            var height =  sizeOfImg.Height;
            var width  =  sizeOfImg.Width;

            if(width == CONFIGURATION.RESIZE_IMAGE.WIDTH)  
            {
                ValueTuple<float, Mat> unProcessed = (1.00f, img);
                return unProcessed;
            }
            
            var scale = (float)(CONFIGURATION.RESIZE_IMAGE.WIDTH / width);
            
            Size dimension = new Size(CONFIGURATION.RESIZE_IMAGE.WIDTH, height*scale);
            
            Cv2.Resize(img, img, dimension);
            
            ValueTuple<float, Mat> resizedImage = (scale, img);
            
            return resizedImage;
        }
}

}