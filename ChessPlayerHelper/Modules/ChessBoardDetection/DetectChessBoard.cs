using NumSharp;
using OpenCvSharp;
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
using core.OpenCvNumSharpConverter; 
using Data;
using System.Linq;
using Aglomera;
using Aglomera.Linkage;
using System.Collections.Generic;
using Dbscan;
using Accord.MachineLearning;
using Accord.Math;
using Dbscan.RBush;
using System.Xaml;

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

        var resizedImg = resizedData.Item2;
    
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
        ValueTuple<NDArray, NDArray> clusters = GetHorizontalAndVerticalLines(detected_lines);
        
        var all_horizontal_lines = clusters.Item1;
        var all_vertical_lines = clusters.Item2;
        var horizontal_lines = EliminateSimilarLines(all_horizontal_lines, all_vertical_lines);
        var vertical_lines = EliminateSimilarLines(all_vertical_lines, all_horizontal_lines);
        var all_intersection_points = _get_intersection_points(horizontal_lines,vertical_lines);
        
        Console.WriteLine("Shape {0} {1} {2}", all_intersection_points.shape[0],all_intersection_points.shape[1],all_intersection_points.shape[2]);
        Mat imgWithPoints = img.Clone();

        for(int i = 0 ; i < all_intersection_points.shape[0]; i++) 
        {
            for(int j = 0; j < all_intersection_points.shape[1]; j++)
            {
                double x = all_intersection_points[i][j][0];
                double y = all_intersection_points[i][j][1];
                Console.WriteLine("{0} {1}", all_intersection_points[i][j][0], all_intersection_points[i][j][1]);

                // Draw a circle for each intersection point
                Cv2.Circle(imgWithPoints, new OpenCvSharp.Point(x, y), 5, new Scalar(0, 255, 0), -1);
            }
            Console.WriteLine("---------------------\n\n");
            // Cv2.ImShow("Image with Points", imgWithPoints);
            // Cv2.WaitKey(0);
        }
        Cv2.ImShow("Image with Points", imgWithPoints);
        Cv2.WaitKey(0);
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
    /**
    * Converts the given degree to radians
    * 
    * Params:
    *   degrees(float) : Degree (0 - 180)
    * 
    * Returns:
    *   float : Returns the radians
    */    
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
    
    public static NDArray sortLines(NDArray lines)
    {
        if (lines.ndim == 0 || lines.shape[0] == 0)
        {
            return lines;
        }

        var rhos = lines[$"...","0"];
        
        // Flatten the array and convert it to a typed array
        var flattenedRhos = rhos.GetData<float>().ToArray();

        // Perform sorting on the flattened array
        var sortedIndices = np.argsort<float>(flattenedRhos);

        return lines[sortedIndices];
    }
    public static NDArray PairwiseDistances(NDArray matrix)
    {
        var numLines = matrix.shape[0];
        var distanceMatrix = np.zeros((numLines, numLines));
        for (int i = 0; i < numLines ; i++)
        {
            for (int j = 0; j < numLines; j++)
            {
                var angleDifference = AbsoluteAngleDifference(matrix[i], matrix[j]);
                distanceMatrix[i][j] = angleDifference;
            }
        }
        return distanceMatrix;
    }
    
    public static NDArray AbsoluteAngleDifference(NDArray x, NDArray y)
    {
        var diff = np.mod(np.abs(x - y), 2 * Math.PI);
        var minDiff = np.min(np.stack(new[] { diff, Math.PI - diff }, axis: -1), axis: -1);
        return minDiff;
    }

  
    class PrecomputedDistanceMetric : IDissimilarityMetric<int>
    {
        private readonly NDArray _distanceMatrix;

        public PrecomputedDistanceMetric(NDArray distanceMatrix)
        {
            _distanceMatrix = distanceMatrix;
        }

        public double Calculate(int instance1, int instance2)
        {
            // Assuming instance1 and instance2 are indices in the distance matrix
            double distance = (double)_distanceMatrix[instance1, instance2];
            return distance;
        }     
    }
    public static ValueTuple<NDArray, NDArray> GetHorizontalAndVerticalLines(NDArray lines)
    {   

        var lines_ = sortLines(lines);

        var thetas = lines_[$"...","1"].reshape(-1, 1);

        var distanceMatrix = PairwiseDistances(thetas);

        double [][] distanceMatrixArray = new double[distanceMatrix.shape[0]][];
        
        for(int i = 0 ; i < distanceMatrix.shape[0]; i++)
            distanceMatrixArray[i] = new double[distanceMatrix.shape[1]];
        
        for(int i = 0; i < distanceMatrix.shape[0]; i++)
            for(int j = 0 ; j < distanceMatrix.shape[1]; j++)
                distanceMatrixArray[i][j] = distanceMatrix[i][j];
         

        KMeans kMeans = new KMeans(k: 2);
        
        int[] kMeansLabels = kMeans.Learn(distanceMatrixArray).Decide(distanceMatrixArray);
        
        NDArray clusters = np.array(kMeansLabels);

        var angle_with_y_axis = AbsoluteAngleDifference(thetas, 0.0);
        
  
        var h_cluster = 0;
        var v_cluster = 0;
        
        if(angle_with_y_axis[clusters == 0].mean() > angle_with_y_axis[clusters == 1].mean())
        {
            h_cluster = 0;
            v_cluster = 1;
        }
        else
        {
            h_cluster = 1;
            v_cluster = 0;
        }
        
        var horizontal_lines = lines_[clusters == h_cluster];
        var vertical_lines = lines_[clusters == v_cluster];
        
        return new ValueTuple<NDArray, NDArray>(horizontal_lines, vertical_lines);
    }
    
public static NDArray get_intersection_points(NDArray rho1, NDArray theta1, NDArray rho2, NDArray theta2)
{
    

    var cos_t1 = np.cos(theta1);
    var cos_t2 = np.cos(theta2);
    var sin_t1 = np.sin(theta1);
    var sin_t2 = np.sin(theta2);
    var x = (sin_t1 * rho2 - sin_t2 * rho1) / (cos_t2 * sin_t1 - cos_t1 * sin_t2);
    var y = (cos_t1 * rho2 - cos_t2 * rho1) / (sin_t2 * cos_t1 - sin_t1 * cos_t2);
 
    var intersections = np.stack(new NDArray[] { x, y }, axis: -1);

    return intersections;
}
public static NDArray _get_intersection_points(NDArray horizontal_lines, NDArray vertical_lines)
{
    
    var horizontalLinesTransposed = np.transpose(horizontal_lines);
    var rho1 = horizontalLinesTransposed[0];
    var theta1 = horizontalLinesTransposed[1];
    
    var verticalLinesTransposed = np.transpose(vertical_lines);
    var rho2 = verticalLinesTransposed[0];
    var theta2 = verticalLinesTransposed[1];
    
   


    int numRows = rho1.shape[0];
    int numCols = rho2.shape[0];

    NDArray rho1_ = np.zeros(new Shape(numRows, numCols));
    NDArray rho2_ = np.zeros(new Shape(numRows, numCols));
    NDArray theta1_ = np.zeros(new Shape(numRows, numCols));
    NDArray theta2_ = np.zeros(new Shape(numRows, numCols));

    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            rho1_[i, j] = rho1[i];
            rho2_[i, j] = rho2[j];
            theta1_[i, j] = theta1[i];
            theta2_[i, j] = theta2[j];
        }
    }

    var intersection_points = get_intersection_points(rho1_, theta1_, rho2_, theta2_);

    return intersection_points;
}   

    public class ClusterPoint : Dbscan.IPointData
    {
        // Implementing the IPointData interface
        public Dbscan.Point Point { get { return new Dbscan.Point(this.x, this.y); } }
        public double x { get; set; }
        public double y { get; set; }

        // Constructor
        public ClusterPoint(double x_, double y_)
        {
            this.x = x_;
            this.y = y_;
        }
    }

    public static NDArray EliminateSimilarLines(NDArray lines, NDArray perpendicular_lines)
    {
        
        var mean = perpendicular_lines.mean(axis: 0, keepdims: true);

        var perp_rho = mean[$":","0"];
        var perp_theta = mean[$":","1"];


        
        var movedAxes = np.moveaxis(lines, -1, 0);
        var rho = movedAxes[$":", "0"];
        var theta = movedAxes[$":", "1"];       

                
        var intersection_points = get_intersection_points(rho, theta, perp_rho, perp_theta);
   
        
  
        List<ClusterPoint> list = new List<ClusterPoint>();        
  
        for(int i = 0 ; i < intersection_points.shape[0]; i++)
        {
            float x = (float)intersection_points[i][0];
            float y = (float)intersection_points[i][1];
            ClusterPoint point = new ClusterPoint(x, y);
            list.Add(point);
        }
        
     
         var epsilon = 12.0;
         var minimumPointsPerCluster = 1;
         var clusters = DbscanRBush.CalculateClusters(list, epsilon, minimumPointsPerCluster);
         var labels = clusters.Clusters;
         var max = clusters.Clusters.Count;
         List<int> cluster_labels = new List<int>();
         List<double[]> clusterPairsList = new List<double[]>();

         for (int i = 0; i < clusters.Clusters.Count; i++)
            {
                var cluster = clusters.Clusters[i];
                foreach (var point in cluster.Objects)
                {
                    cluster_labels.Add(i);
                }
            }
        var clusters_ndarray = np.array(cluster_labels);

    
        NDArray clusterPairs = np.zeros((max, 2));
        for (int i = 0 ; i < max; i++)
        {
            var lines_in_cluster = lines[clusters_ndarray == i];
            var rho_ = lines_in_cluster[$"...","0"];

            var sortedRho = rho_.Data<double>().OrderBy(x => x).ToArray();
            
            int medianIndex = sortedRho.Length / 2;
            
            clusterPairs[i] = lines_in_cluster[medianIndex];

        }
        
        return clusterPairs;
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

/*




*/

