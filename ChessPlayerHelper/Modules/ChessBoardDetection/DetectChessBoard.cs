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
        //var all_intersection_points = _get_intersection_points(horizontal_lines,vertical_lines);
        double[][][] data = new double[][][]
        {
            new double[][]
            {
                new double[] {272.6676, 120.62514},
                new double[] {310.5376, 120.294655},
                new double[] {387.1103, 119.62642},
                new double[] {462.05618, 118.97237},
                new double[] {538.1382, 118.30841},
                new double[] {612.99664, 117.65513},
                new double[] {683.80255, 117.03722},
                new double[] {758.7367, 116.38327},
                new double[] {834.3823, 115.723114},
                new double[] {909.9195, 115.06392},
                new double[] {955.67737, 114.6646},
                new double[] {1010.60596, 114.18524},
            },
            new double[][]
            {
                new double[] {260.79944, 150.00002},
                new double[] {299.4312, 150.00002},
                new double[] {378.68692, 150.00002},
                new double[] {456.30554, 150.00002},
                new double[] {535.08655, 150.00003},
                new double[] {612.71436, 150.00003},
                new double[] {686.68646, 150.00003},
                new double[] {764.66425, 150.00003},
                new double[] {843.5668, 150.00005},
                new double[] {922.98157, 150.00003},
                new double[] {968.8888, 150.00003},
                new double[] {1027.6887, 150.00005},
            },
            new double[][]
            {
                new double[] {238.60803, 204.92558},
                new double[] {279.02725, 204.57285},
                new double[] {363.75763, 203.83342},
                new double[] {446.46188, 203.11166},
                new double[] {530.0427, 202.38226},
                new double[] {612.2635, 201.66472},
                new double[] {691.1463, 200.97632},
                new double[] {773.52594, 200.2574},
                new double[] {856.8384, 199.53035},
                new double[] {941.2249, 198.79391},
                new double[] {986.9827, 198.39459},
                new double[] {1050.5073, 197.84023},
            },
            new double[][]
            {
                new double[] {219.18472, 253.00002},
                new double[] {260.92108, 253.00002},
                new double[] {350.12247, 253.00003},
                new double[] {437.21564, 253.00002},
                new double[] {525.1688, 253.00003},
                new double[] {611.8155, 253.00003},
                new double[] {695.6978, 253.00003},
                new double[] {782.8259, 253.00003},
                new double[] {871.1656, 253.00003},
                new double[] {961.4917, 253.00003},
                new double[] {1007.39886, 253.00003},
                new double[] {1076.8173, 253.00005},
            },
            new double[][]
            {
                new double[] {194.94315, 313},
                new double[] {238.488, 313},
                new double[] {333.483, 313.00003},
                new double[] {426.09528, 313.00003},
                new double[] {519.3915, 313.00003},
                new double[] {611.2919, 313.00003},
                new double[] {700.94714, 313.00003},
                new double[] {793.4056, 313.00003},
                new double[] {887.24255, 313.00003},
                new double[] {983.92474, 313.00003},
                new double[] {1029.832, 313.00003},
                new double[] {1105.4358, 313.00003},
            },
            new double[][]
            {
                new double[] {167.06534, 382},
                new double[] {212.68997, 382},
                new double[] {314.34763, 382.00006},
                new double[] {413.30692, 382.00003},
                new double[] {512.7475, 382.00003},
                new double[] {610.6897, 382.00003},
                new double[] {706.9839, 382.00003},
                new double[] {805.57214, 382.00003},
                new double[] {905.7311, 382.00003},
                new double[] {1009.72284, 382.00006},
                new double[] {1055.63, 382.00006},
                new double[] {1138.3472, 382.00003},
            },
            new double[][]
            {
                new double[] {137.48372, 455.21707},
                new double[] {185.15964, 455.63318},
                new double[] {293.66473, 456.58005},
                new double[] {399.31342, 457.50204},
                new double[] {505.3884, 458.42776},
                new double[] {610.0148, 459.3408},
                new double[] {713.8296, 460.2468},
                new double[] {819.5318, 461.16922},
                new double[] {927.19617, 462.10883},
                new double[] {1040.0425, 463.0936},
                new double[] {1086.1, 463.49554},
                new double[] {1177.5994, 464.29407},
            },
            new double[][]
            {
                new double[] {100.40101, 547},
                new double[] {150.999, 547},
                new double[] {268.58908, 547},
                new double[] {382.72595, 547},
                new double[] {496.85983, 547},
                new double[] {609.24976,547},
                new double[] {721.41956, 547},
                new double[] {834.66614, 547.00006},
                new double[] {949.9427, 547.00006},
                new double[] {1071.4138, 547.00006},
                new double[] {1117.321, 547.00006},
                new double[] {1217.0481, 547.00006},
            },
             new double[][]
            {
                new double[] {58.972416, 649.53937},
                new double[] {112.48648, 650.0064},
                new double[] {239.71497, 651.1167},
                new double[] {363.22925, 652.1945},
                new double[] {486.62708, 653.2715},
                new double[] {608.3131, 654.3334},
                new double[] {730.9036, 655.40326},
                new double[] {853.9699, 656.4772},
                new double[] {979.5708, 657.5733},
                new double[] {1113.1914, 658.7394},
                new double[] {1159.249, 659.14136},
                new double[] {1271.002, 660.1166},
            }
        };
          Mat imgWithPoints = img.Clone();
        foreach (var pointsSet in data)
        {
            foreach (var point in pointsSet)
            {
                int x = (int)point[0];
                int y = (int)point[1];

                // Draw a circle for each intersection point
                Cv2.Circle(imgWithPoints, new OpenCvSharp.Point(x, y), 5, new Scalar(0, 255, 0), -1);
            }
        }

        // Show the image with intersection points
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
// public static NDArray _get_intersection_points(NDArray horizontal_lines, NDArray vertical_lines)
// {
//     var Mean1 = np.moveaxis(horizontal_lines, -1, 0);
//     var Mean2 = np.moveaxis(vertical_lines, -1, 0);

//     var rho1 = Mean1[0];
//     var theta1 = Mean1[1];

//     var rho2 = Mean2[0];
//     var theta2 = Mean2[1];
//     Kwargs kwargs = new Kwargs("ij", true, false);
//     var MeshGrid1 = np.meshgrid(rho1, rho2, kwargs);
//     var MeshGrid2 =  np.meshgrid(theta1, theta2, kwargs);
    
//     var rho1_  = MeshGrid1.Item1;
//     var rho2_ = MeshGrid1.Item2;
//     var theta1_ = MeshGrid2.Item1;
//     var theta2_ = MeshGrid2.Item2;

//     var intersection_points = get_intersection_points(rho1_, theta1_, rho2_, theta2_);
//     for(int i = 0 ; i < intersection_points.shape[0]; i++)
//         for(int j = 0 ; j < intersection_points.shape[1];j++)
//         Console.WriteLine("{0}{1}", intersection_points[i][j][0], intersection_points[i][j][1]);
    
    
//     return horizontal_lines;
// }   

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

