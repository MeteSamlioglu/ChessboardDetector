using NumSharp;
using OpenCvSharp;
using System;
using System.Runtime.InteropServices;
using System.Diagnostics;
static class DetectCorners
{
   public static NDArray Corner(Mat img)
    {
        
        NDArray npImage = MatToNDArray(img);

        ValueTuple<float, Mat> resizedData = resize_image(npImage, img);
                        
        return MatToNDArray(resizedData.Item2);
    }

    public static ValueTuple<float, Mat> resize_image(NDArray npImage, Mat img)
    {        
        int height =  npImage.shape[0];
        int width  =  npImage.shape[1];
        
        Console.WriteLine("Width : {0}, Height : {1}", width, height);

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
    
    
    static NDArray MatToNDArray(Mat mat)
    {
        byte[] pixelData = new byte[mat.Total() * mat.Channels()];
        Marshal.Copy(mat.Data, pixelData, 0, pixelData.Length);

        NDArray ndArray = np.array(pixelData).reshape(mat.Rows, mat.Cols, mat.Channels());
        return ndArray;
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