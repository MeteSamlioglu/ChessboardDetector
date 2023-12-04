using NumSharp;
using OpenCvSharp;
using System.Runtime.InteropServices;
using System;
using System.Linq;
namespace core.OpenCvNumSharpConverter
{

    static class MatArrayConverter
    {
        public static NDArray MatToNDArray(Mat mat)
        {
            byte[] pixelData = new byte[mat.Total() * mat.Channels()];
            Marshal.Copy(mat.Data, pixelData, 0, pixelData.Length);

            NDArray ndArray = np.array(pixelData).reshape(mat.Rows, mat.Cols, mat.Channels());            
                    
            return ndArray;
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
            // Convert NDArray to flattened byte array
            var flatArray = ndArray.GetData<byte>().ToArray(); // Convert to byte array

            // Create Mat from pixel data    
            Mat mat = new Mat(ndArray.shape[0], ndArray.shape[1], MatType.CV_8UC(ndArray.shape[2]), flatArray);
            
            return mat;
        }
    }






}