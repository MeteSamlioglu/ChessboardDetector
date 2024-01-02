using NumSharp;
using OpenCvSharp;
using System.Runtime.InteropServices;
using System;
using System.Linq;
using System.Drawing.Drawing2D;
using MathNet.Numerics.LinearAlgebra;

namespace core.OpenCvNumSharpConverter
{

static class MatArrayConverter
{
    public static NDArray MatToNDArray(Mat transformation_matrix)
    {
        
        NDArray transformation_matrix_ = np.zeros((transformation_matrix.Rows, transformation_matrix.Cols));
        for (int i = 0; i < transformation_matrix.Rows; i++)
        {
            for (int j = 0; j < transformation_matrix.Cols; j++)
            {
                transformation_matrix_[i][j] = transformation_matrix.At<double>(i,j);

            }
        }
        return transformation_matrix_;
    }
        
        public static NDArray MatToNDArray(LineSegmentPolar[] Mat)
        {
            var linesArray = Mat
            .Select(line => new float[] { line.Rho, line.Theta})
            .ToArray();
            return np.array(linesArray);
        }

    
    public static Mat NDArrayToMat(NDArray ndArray)
    {
        if (ndArray.ndim != 2)
        {
            throw new ArgumentException("NDArray must be two-dimensional.");
        }

        var shape = ndArray.shape;
        var result = new Mat(shape[0], shape[1], MatType.CV_64F); // Adjust MatType as needed

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                double value = Convert.ToDouble(ndArray[i, j].ToString()); // Convert the value to double
                result.Set(i, j, value);
            }
        }

        return result;
    }
        // Helper method to check if NDArray has numeric data


    public static Matrix<double> NDArrayToMatrix(NDArray src)
    {
        var row_count = src.shape[0];
        var col_count = src.shape[1];
        var matrix = Matrix<double>.Build.Dense(row_count, col_count);

        for (int i = 0; i < row_count; i++)
        {
            for (int j = 0; j < col_count; j++)
            {
                matrix[i, j] = src[i, j];
            }
        }   

        return matrix;
    }
    public static NDArray MatrixToNDArray(Matrix<double> src)
    {
        var rows_ = src.RowCount;
        var cols_ = src.ColumnCount;
        var inverse_matrix = np.zeros(new Shape(rows_, cols_));

        for (int i = 0; i < rows_; i++)
        {
            for (int j = 0; j < cols_; j++)
            {
                inverse_matrix[i, j] = src[i, j];
            }
        }
        return inverse_matrix;
    }
}






}