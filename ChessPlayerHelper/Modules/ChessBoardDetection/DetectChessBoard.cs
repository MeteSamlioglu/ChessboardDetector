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
using Microsoft.VisualBasic.Devices;
using MathNet.Numerics.LinearAlgebra;

namespace Modules
{
    
    /**
    * Static class responsible for detecting the four corners/borders of a chessboard in an image.
    * Contains methods for edge detection and resizing to assist in chessboard corner detection.
    */
    
static class DetectChessBoard
{    
    
    public static NDArray FindCorners(Mat img_)
    {
    /**
        Find the four corner points of the chessboard for an image.
        
        Params:
            img(Mat): The image that is taken from the frame(as a MatLike)
        Returns:
            NDArray: That contains the coordinates of the four corners 
    **/
        Mat img = new Mat();
        Cv2.CvtColor(img_, img, ColorConversionCodes.BGR2RGB);
     
        ValueTuple<float, Mat> resizedData = ResizeImage(img);
        var img_scale = resizedData.Item1;
        var resizedImg = resizedData.Item2;
    
        // Convert from BGR to Grayscale
        Mat gray = new Mat(); 
        
        Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
    
        var edges_detected = DetectEdges(gray,CONFIGURATION.EDGE_DETECTION.LOW_THRESHOLD,
                                              CONFIGURATION.EDGE_DETECTION.HIGH_THRESHOLD,
                                              CONFIGURATION.EDGE_DETECTION.APERTURE);        
        
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

        //var best_configuration = (ValueTuple<ValueTuple<int, int, int, int>, NDArray, NDArray, NDArray, NDArray>?)null;
        
        var best_configuration = (
            (0, 0, 0, 0),    
            default(NDArray),
            default(NDArray), 
            default(NDArray), 
            default(NDArray)  
        );        

        int best_num_inliers = 0;
        int iterations = 0;
        while (iterations < 200 || best_num_inliers < 30)
        {   
            var rows = ChooseFromRange(horizontal_lines.shape[0]);
            var columns = ChooseFromRange(vertical_lines.shape[0]);
            var row1 = rows[0];
            var row2 = rows[1];
            var col1 = columns[0];
            var col2 = columns[1];
            
            var transformation_matrix = ComputeHomography(all_intersection_points,row1,row2,col1,col2);
            var warped_points = WarpPoints(transformation_matrix, all_intersection_points);
            var outliers = DiscardOutliers(warped_points,all_intersection_points);
            
            var warped_points_ = outliers.Item1;
            
            var intersection_points_ = outliers.Item2;
            
            var horizontal_scale = (int)outliers.Item3;
            
            var vertical_scale = (int)outliers.Item4;     
            
            var reshapedShape = np.array(new int[] { warped_points_.shape[0],warped_points_.shape[1]});
            
            int num_inliers = np.prod(reshapedShape);
            
            if(num_inliers > best_num_inliers)
            {
             
                var multArray = NumSharpMethods.MultNDArray(warped_points_, horizontal_scale, vertical_scale);
     
                var Configuration = QuantizePoints(warped_points_,intersection_points_);
                int xmin = Configuration.Item1.Item1;
                int xmax = Configuration.Item1.Item2;
                int ymin = Configuration.Item1.Item3;
                int ymax = Configuration.Item1.Item4;
                var scale = Configuration.Item2;
                var scaled_quantized_points = Configuration.Item3;
                var IntersectionPoints = Configuration.Item4;
                var warped_img_size = Configuration.Item5;
                var reshapedQuantizedPoints = np.array(new int[] {scaled_quantized_points.shape[0],scaled_quantized_points.shape[1]});
                num_inliers = np.prod(reshapedQuantizedPoints);
                //Console.WriteLine("num_inliers {0}",num_inliers);
                if(num_inliers > best_num_inliers)
                {
                    best_num_inliers = num_inliers;
                    best_configuration = Configuration;
                }
            }
           
            iterations =  iterations + 1;
            if(iterations > 10000)
            {
                Console.WriteLine("Chessboard not located Exception");
            }
        }
        var points = best_configuration.Item1;
        int xmin_ = points.Item1;
        int xmax_ = points.Item2;
        int ymin_ = points.Item3;
        int ymax_ = points.Item4;
        var scale_ = best_configuration.Item2;
        var quantized_points_ = best_configuration.Item3;
        var intersection_points = best_configuration.Item4;
        var warped_img_size_ = best_configuration.Item5;

    
        
        var transformation_matrix_ = ComputeTransformationMatrix(intersection_points, quantized_points_);

        var matrix_format = MatArrayConverter.NDArrayToMatrix(transformation_matrix_);
  
        // Calculate the inverse transformation matrix
        var inverseTransformationMatrix = matrix_format.Inverse();
 
        var inverse_matrix = MatArrayConverter.MatrixToNDArray(inverseTransformationMatrix);

        var width = warped_img_size_[0].astype(np.int32);
        
        var height = warped_img_size_[1].astype(np.int32);        
        
        var dims = new Size(width, height);
        
        var transformed_img = MatArrayConverter.NDArrayToMat(transformation_matrix_);
        
        var warped = new Mat(dims, img.Type());

        Cv2.WarpPerspective(img,warped, transformed_img, dims);
        
        // Cv2.ImShow("Warped",warped);
        // Cv2.WaitKey(0);
        
        var warpedNDArray = MatArrayConverter.MatToNDArray(warped);

        var gray_ = MatArrayConverter.MatToNDArray(gray);
        
        var borders = np.zeros_like(gray_);
        
        borders[$"3:-3",$"3:-3"] = 1;
        var borders_ = MatArrayConverter.NDArrayToMat(borders);
        var warped_borders = new Mat();
        
        Cv2.WarpPerspective(borders_,warped_borders,transformed_img,dims);
        // Cv2.ImShow("Warped2",warped_borders);
        // Cv2.WaitKey(0);
        
        var warped_borders_ = MatArrayConverter.MatToNDArray(warped_borders); 
        
        var warped_mask = warped_borders_ == 1;

        // var VerticalBorders = ComputeVerticalBorders(warped, warped_mask,scale_, xmin_, xmax_); 
        // var Xmin = VerticalBorders.Item1;
        // var Xmax = VerticalBorders.Item2;
        
        // var scaled_xmin = Xmin * (int)scale_[0];
        // var scaled_xmax = Xmax * (int)scale_[0];
        
        // Console.WriteLine("scaled_xmin {0} scaled_xmax {1}",scaled_xmin, scaled_xmax);
        var corners = np.array([[xmin_, ymin_],
                                [xmax_, ymin_],
                                [xmax_, ymax_],
                                [xmin_, ymax_]]).astype(np.float32);
        corners = corners * scale_;
        
        var img_corners = WarpPoints_(inverse_matrix, corners);
        
        img_corners = img_corners / 1;
        
        Console.WriteLine($"{img_corners}");
        
        for(int i = 0 ; i < img_corners.shape[0]; i++)
        {
            // Use alias for OpenCvSharp.Point
            Console.WriteLine("point {0} {1}",img_corners[i][0], img_corners[i][1]);
            var center = new OpenCvSharp.Point((double)img_corners[i][0], (double)img_corners[i][1]);
            Cv2.Circle(img, center, 5, Scalar.Red, -1);
        }
        Cv2.ImShow("Points", img);
        Cv2.WaitKey(0);
        
        return img_corners;
    }
    public static NDArray get_nonmax_supressed(NDArray G_x, int x, NDArray scale)
    {
        x = (x*scale[0]).astype(np.int32);
        var thresh = CONFIGURATION.BORDER_REFINEMENT.LINE_WIDTH / 2;
        var slicedArray = G_x[":", x - thresh, x + thresh + 1];

        return slicedArray.max(axis : 1);
    }
    public static ValueTuple<int, int>ComputeVerticalBorders(Mat warped, NDArray warped_mask, NDArray scale, int xmin, int xmax)
    {
        Mat sobelImg = new Mat();

        Cv2.Sobel(warped, sobelImg, MatType.CV_64F, 1, 0, CONFIGURATION.BORDER_REFINEMENT.SOBEL_KERNEL_SIZE);
        // Cv2.ImShow("Sobel",sobelImg);
        // Cv2.WaitKey(0);        
        var sobelArray = MatArrayConverter.MatToNDArray(sobelImg);
        
        var G_x = np.abs(sobelArray);
        
        var newValues = G_x * warped_mask.astype(NPTypeCode.Int32);

        
        //G_x[~warped_mask] = 0;
        
        var G_x_ = MatArrayConverter.NDArrayToMat(newValues);
        
        var res = DetectEdges(G_x_, CONFIGURATION.BORDER_REFINEMENT.EDGE_DETECTION.VERTICAL.LOW_THRESHOLD,
                                    CONFIGURATION.BORDER_REFINEMENT.EDGE_DETECTION.VERTICAL.HIGH_THRESHOLD,
                                    CONFIGURATION.BORDER_REFINEMENT.EDGE_DETECTION.VERTICAL.APERTURE);
        Cv2.ImShow("Res",res);
        Cv2.WaitKey(0);
        G_x = MatArrayConverter.MatToNDArray(G_x_);
        G_x = G_x * warped_mask.astype(NPTypeCode.Int32);

        // G_x[~warped_mask] = 0;

        while(xmax - xmin < 8)
        {
            Console.WriteLine("Alo1");
            var top = get_nonmax_supressed(G_x, xmax + 1, scale);
            var bottom = get_nonmax_supressed(G_x, xmin - 1, scale);
            if(top.sum() > bottom.sum())
            {
                xmax+=1;
            }
            else
            {
                xmin -= 1;
            }
        }

        return new ValueTuple<int, int>(xmin,xmax);
    }
    public static ValueTuple<ValueTuple<int, int, int,int>, NDArray,NDArray,NDArray,NDArray>QuantizePoints(NDArray warped_scaled_points, NDArray intersection_points)
    {

        var mean_col_xs = warped_scaled_points[$"...", 0].mean(axis : 0);
        var mean_row_ys = warped_scaled_points[$"...", 1].mean(axis : 1);

        // Console.WriteLine($"{warped_scaled_points}");
        // Console.WriteLine("mean_col_xs {0}, Type {1}",mean_col_xs, mean_col_xs.dtype);

         var col_xs = np.round_(mean_col_xs).astype(np.int32);
         var row_ys = np.round_(mean_row_ys).astype(np.int32);

        // Console.WriteLine($"{col_xs}");  
        
        // Console.WriteLine($"{row_ys}");  

        var UniqueRes = NumSharpMethods.Unique(col_xs, row_ys);
        var col_xs_ = UniqueRes.Item1;
        var col_indices = UniqueRes.Item2;
        var row_ys_ =  UniqueRes.Item3;
        var row_indices = UniqueRes.Item4;

        var intersection_points_ = NumSharpMethods.SliceIntegerNDArray(intersection_points,row_indices, col_indices);
        

        int xmin = col_xs_.min().astype(NPTypeCode.Int32).GetData<int>()[0];;
        int xmax = col_xs_.max().astype(NPTypeCode.Int32).GetData<int>()[0];
        int ymin = row_ys_.min().astype(NPTypeCode.Int32).GetData<int>()[0];
        int ymax = row_ys.max().astype(NPTypeCode.Int32).GetData<int>()[0];

       //Ensure we a have a maximum of 9 rows/cols
        while( (xmax - xmin) > 9)
        {
            xmax -= 1;
            xmin += 1;
        }
        while ((ymax - ymin) > 9)
        {
            ymax -= 1;
            ymin += 1;
        }
        
        var col_mask = NumSharpMethods.Mask(col_xs_,xmin, xmax);
        var row_mask = NumSharpMethods.Mask(row_ys_,xmin, xmax);
        
        var colXs = NumSharpMethods.Slice1DBoolean(col_xs_,col_mask);
 
        var rowYs = NumSharpMethods.Slice1DBoolean(row_ys_,row_mask);

        
        var intersection_points_mask = NumSharpMethods.SliceNDArray(intersection_points_,row_mask, col_mask, 1);

        var quantized_points = NumSharpMethods.MeshGrid(colXs, rowYs, -1);
   
        var xmin_ = - xmin + CONFIGURATION.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG;
        var ymin_ = - ymin + CONFIGURATION.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG;
        
        var translation = np.array([xmin_, ymin_]);
        
        var scale = np.array(CONFIGURATION.BORDER_REFINEMENT.WARPED_SQUARE_SIZE);
        
        NDArray scaled_quantized_points;

        if ((quantized_points.size == 0) || (translation.size == 0))
            scaled_quantized_points = np.empty(new Shape(0, 1));
        
        else
            scaled_quantized_points = (quantized_points + translation) * scale;
        
        var Min = np.array(xmin, ymin) + translation;
        var Max = np.array(xmax, ymax) + translation;
        
        xmin = Min[0];
        ymin = Min[1];
        xmax = Max[0];
        ymax = Max[1];

        
        var warped_img_size = (np.array(xmax, ymax) + CONFIGURATION.BORDER_REFINEMENT.NUM_SURROUNDING_SQUARES_IN_WARPED_IMG) * scale;
        
        
        ValueTuple<int,int,int,int> CornerLimits = new ValueTuple<int,int,int,int>(xmin,xmax,ymin,ymax); 
        
        return new ValueTuple<ValueTuple<int,int,int,int>,NDArray, NDArray,NDArray,NDArray>(CornerLimits,scale,scaled_quantized_points,intersection_points_mask, warped_img_size);
    }
    public static ValueTuple<NDArray,NDArray>FindBestScale(NDArray values)
    {
        NDArray scales = np.arange(1,9);
         
        var indices = np.argsort<double>(scales);
         
        NDArray sortedArray = scales[indices];

        var scaled_values = np.expand_dims(values, axis : -1) * sortedArray;
        
        var diff = np.abs(np.round_(scaled_values) - scaled_values);
        
        var inlier_mask = diff < CONFIGURATION.RANSAC.OFFSET_TOLERANCE / scales;

        var numInliers = CalculateNumInliers(inlier_mask);

        var bestNumInliers = np.max(numInliers);
        
        var threshold = (1 - CONFIGURATION.RANSAC.BEST_SOLUTION_TOLERANCE) * bestNumInliers;
        int index = 0;
        int maxValue = numInliers[0];

        for(int i = 0 ; i < numInliers.shape[0]; i++)
        {
            int numInlierValue = numInliers[i];
            double thresholdValue = (double)threshold;
            if (numInlierValue > thresholdValue)
            {
                maxValue = numInlierValue;
                index = i;
                break;
            }
        }
        return new ValueTuple<NDArray, NDArray>(sortedArray[index], inlier_mask[$"...", index]);
    }
    
    public static int ArgMax(NDArray array)
    {
        if (array.size == 0)
            throw new ArgumentException("Input array must not be empty.");

        int argmax = 0;
        dynamic maxVal = array[0];

        for (int i = 1; i < array.size; i++)
        {
            dynamic currentVal = array[i];
            if (currentVal > maxVal)
            {
                maxVal = currentVal;
                argmax = i;
            }
        }
        return argmax;
    }

    public static NDArray CalculateNumInliers(NDArray inlierMask)
    {
        int numMatrices = inlierMask.shape[inlierMask.ndim - 1];
        int[] numInliers = new int[numMatrices];

        for (int k = 0; k < numMatrices; k++)
        {
            int sum = 0;
            for (int i = 0; i < inlierMask.shape[0]; i++)
            {
                for (int j = 0; j < inlierMask.shape[1]; j++)
                {
                    sum += inlierMask[i, j, k].Data<bool>()[0] ? 1 : 0;
                }
            }
            numInliers[k] = sum;
        }

        return np.array(numInliers);
    }
    public static ValueTuple<NDArray, NDArray, double, double> DiscardOutliers(NDArray warped_points, NDArray all_intersection_points)
    {
        var horizontal = FindBestScale(warped_points[$"...","0"]);
        var vertical = FindBestScale(warped_points[$"...", "1"]);
        
        var horizontal_scale = horizontal.Item1;
        var horizontal_mask = horizontal.Item2;
        var vertical_scale = vertical.Item1;
        var vertical_mask = vertical.Item2;
        var mask = np.logical_and(horizontal_mask, vertical_mask);

        /* Keep rows / cols that have more than %50 inliers */
        NDArray rows_to_consider = mask[$":",":","-1"];
        
        var num_of_rows_to_consider = NumSharpMethods.AnySum(rows_to_consider, -1);
        var num_of_columns_to_consider = NumSharpMethods.AnySum(mask, -2);
        
        var rows_to_keep = (NumSharpMethods.Sum(mask, -1) / (double)num_of_rows_to_consider) > CONFIGURATION.MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE;
        
        var columns_to_keep = (NumSharpMethods.Sum(mask, -2) / (double)num_of_columns_to_consider) > CONFIGURATION.MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE;
        
        var WarpedPoints = NumSharpMethods.SliceNDArray(warped_points,rows_to_keep,columns_to_keep);
        
        var IntersectionPoints =  NumSharpMethods.SliceNDArray(all_intersection_points, rows_to_keep, columns_to_keep);
    
        return new ValueTuple<NDArray, NDArray,int,int>(WarpedPoints, IntersectionPoints, horizontal_scale, vertical_scale);
    }

    public static NDArray ChooseFromRange(int upper_bound, int n = 2)
    {
        var indices = np.arange(upper_bound);
        var n_ = new Shape(n);

        // Set a seed for reproducibility
        np.random.seed(42);

        // Fisher-Yates shuffle
        var rng = new Random();
        var shuffledIndices = indices.ToArray<int>().OrderBy(x => rng.Next()).ToArray();

        // Take the first n elements
        var selectedIndices = shuffledIndices.Take(n).OrderBy(x => x).ToArray();

        return np.array(selectedIndices);
    }
    public static NDArray ComputeHomography(NDArray intersection_points, int row1, int row2, int col1, int col2)
    {
        var p1 = intersection_points[row1, col1]; //Top Left
        var p2 = intersection_points[row1, col2]; //Top Right
        var p3 = intersection_points[row2, col2];  //Bottom Right
        var p4 = intersection_points[row2, col1];  // Bottom Left
        
        var src_points = np.stack([p1, p2, p3, p4]);
        var dst_points = np.array([[0, 0], //top left
                           [1, 0], //top right
                           [1, 1],  //bottom right
                           [0, 1]]); // bottom left

 
        return ComputeTransformationMatrix(src_points, dst_points);
    }
    public static NDArray ComputeTransformationMatrix(NDArray src_points, NDArray dst_points)
    {
        var source_points = src_points.reshape(-1,2);
        var destination_points = dst_points.reshape(-1, 2);
        
     
        Mat sourceMat = MatArrayConverter.NDArrayToMat(source_points);
        Mat destinationMat = MatArrayConverter.NDArrayToMat(destination_points);

        var transformation_matrix = Cv2.FindHomography(sourceMat, destinationMat);
        var transformation_matrix_ = MatArrayConverter.MatToNDArray(transformation_matrix);

        return transformation_matrix_;
    }
    
    public static NDArray WarpPoints(NDArray TransformationMatrix, NDArray Points)
    {
        var points = CoordinatesConverter.ToHomogenousCoordinates(Points);
        var transposedMatrix = np.transpose(TransformationMatrix);
        
        int n = points.shape[0];
        int m = points.shape[1];
        int p = transposedMatrix.shape[1];

        NDArray warpedPoints = np.zeros(n, m, p);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < m; j++)
            {
                for (int k = 0; k < p; k++)
                {
                    warpedPoints[i, j, k] = 0;

                    for (int l = 0; l < transposedMatrix.shape[0]; l++)
                    {
                        warpedPoints[i, j, k] += points[i, j, l] * transposedMatrix[l, k];
                    }
                }
            }
        }
        
        var HomogenousCoordinates = CoordinatesConverter.FromHomogeneousCoordinates(warpedPoints);
   
   
        return HomogenousCoordinates;
    }
   
    public static NDArray WarpPoints_(NDArray TransformationMatrix, NDArray Points)
    {
        var points = CoordinatesConverter.ToHomogenousCoordinates(Points);
        var transposedMatrix = np.transpose(TransformationMatrix);

        // Ensure the last dimension is 3
        if (points.shape.Length < 2 || points.shape[points.shape.Length - 1] != 3)
        {
            throw new ArgumentException("Invalid shape for homogeneous coordinates");
        }

        NDArray warpedPoints = np.dot(points, transposedMatrix);
        Console.WriteLine($"{warpedPoints}");
        NDArray lastCoordinate = warpedPoints[":", "-1"];

        NDArray result = warpedPoints[":", ":-1"] / np.expand_dims(lastCoordinate, axis: -1);
        
        return result;
    }

    public static Mat DetectEdges(Mat grayImg, int Lt, int Ht, int aperture)
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
        int LOW_THRESHOLD = Lt;
        int HIGH_THRESHOLD = Ht;
        int APERTURE = aperture;
       
         if (grayImg.Depth() != MatType.CV_8U)
        {
            grayImg.ConvertTo(grayImg, MatType.CV_8U);
        }
        if (grayImg.Type().Channels != 1)
        {
            // Normalize and convert to UInt8
            Cv2.Normalize(grayImg, grayImg, 0, 255, NormTypes.MinMax, MatType.CV_8U);
        }
        
        Mat Edges = new Mat(grayImg.Size(), grayImg.Type());
        
        Cv2.Canny(grayImg,  Edges, LOW_THRESHOLD, HIGH_THRESHOLD, APERTURE);
        
        
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
        var rho = movedAxes[":", "0"];
        var theta = movedAxes[":", "1"];       
        
        var intersection_points = get_intersection_points(rho, theta, perp_rho, perp_theta);
   
        
        List<ClusterPoint> list = new List<ClusterPoint>();        
  
        for(int i = 0 ; i < intersection_points.shape[0]; i++)
        {
            float x = (float)intersection_points[i][0];
            float y = (float)intersection_points[i][1];
            ClusterPoint point = new ClusterPoint(x, y);
            list.Add(point);
        }
     
         var epsilon = 12;
         var minimumPointsPerCluster = 1;
         var clusters = DbscanRBush.CalculateClusters(list, epsilon, minimumPointsPerCluster);
         var labels = clusters.Clusters;
         var max = clusters.Clusters.Count;
         List<int> cluster_labels = new List<int>();
         Dictionary<ClusterPoint, int> pointToClusterMap = new Dictionary<ClusterPoint, int>();


        for (int i = 0; i < clusters.Clusters.Count; i++)
        {
            var cluster = clusters.Clusters[i];
            foreach (var point in cluster.Objects)
            {
                pointToClusterMap.Add(point, i);
            }
        }
        foreach (var clusterPoint in list)
        {
             ClusterPoint findPoint = clusterPoint;
            if (pointToClusterMap.TryGetValue(findPoint, out int clusterLabel))
            {
                cluster_labels.Add(clusterLabel);
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
