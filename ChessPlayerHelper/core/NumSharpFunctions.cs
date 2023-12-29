using NumSharp;
using OpenCvSharp;
using System.Runtime.InteropServices;
using System;
using System.Linq;
using System.Collections.Generic;

namespace core.OpenCvNumSharpConverter
{
    static class NumSharpMethods
    {   
        
        public static int AnySum(NDArray SlicedNDArray, int axis)
        {
            int AnySumResult = 0;
            
            switch(axis)
            {
                case -1:
                    NDArray mask_rows = np.zeros((SlicedNDArray.shape[0],1));

                    for(int i = 0 ; i < SlicedNDArray.shape[0]; i++)
                    {
                        mask_rows[i] = np.any(SlicedNDArray[i]);
                    }
                    
                    int rows_to_consider = 0;
                    
                    for(int i = 0 ; i < mask_rows.shape[0]; i++)
                    {
                        if(mask_rows[i][0] == true)
                        {
                            rows_to_consider++;
                        }
                    }
                    AnySumResult = rows_to_consider;

                break;
                case -2:
                    
                    List<List<bool>> groupedColumns = new List<List<bool>>();
                    
                    for (int col = 0; col < SlicedNDArray.shape[1]; col++)
                    {
                        List<bool> columnValues = new List<bool>();

                        for (int row = 0; row < SlicedNDArray.shape[0]; row++)
                        {
                            columnValues.Add(SlicedNDArray[row, col]);
                        }
                        
                        groupedColumns.Add(columnValues);
                    }

                    int columns_to_consider = 0;
                    
                    NDArray mask_columns = np.zeros((groupedColumns.Count,1)); 
                    
                    for(int i = 0 ; i < groupedColumns.Count; i++)
                    {
                        List<bool> columnValue = groupedColumns[i];
                        mask_columns[i] = np.any(np.array(columnValue));
                        if(mask_columns[i][0] == true)
                        {
                            columns_to_consider++;
                        }
                    }  
                    AnySumResult= columns_to_consider;
                break;
            }
            
            return AnySumResult;
        }
        
        public static NDArray Sum(NDArray SlicedNDArray, int axis)
        {
             NDArray SumResult = np.zeros((SlicedNDArray.shape[0],1));
            
            switch(axis)
            {
                case -1:

                    for(int i = 0 ; i < SlicedNDArray.shape[0]; i++)
                    {
                        int res = 0;
                        for(int j = 0; j < SlicedNDArray.shape[1]; j++)
                        {
                            if(SlicedNDArray[i][j] == true)
                            {
                               res++;
                            }
                        }
                        SumResult[i] = res;
                    }
                break;
                case -2:     
                    int numCols = SlicedNDArray.shape[1];
                    SumResult = np.zeros(new Shape(numCols, 1));

                    for (int col = 0; col < numCols; col++)
                    {
                        int res_column = 0;
                        for (int row = 0; row < SlicedNDArray.shape[0]; row++)
                        {
                            if (SlicedNDArray[row, col] == true)
                            {
                                res_column++;
                            }
                        }
                        SumResult[col] = res_column;
                    }
                break;                 
            }
            
            return SumResult;
        }
    
        public static NDArray SliceNDArray(NDArray arr, NDArray rows_to_slice, NDArray cols_to_slice )
        {
            List<int> rowIndices = new List<int>();
            List<int> columnIndices = new List<int>();
            
            for(int i = 0 ; i < rows_to_slice.shape[0]; i++)
                if(rows_to_slice[i][0])
                     rowIndices.Add(i);
            
            for(int i = 0 ; i < cols_to_slice.shape[0]; i++)
                if(cols_to_slice[i][0])
                    columnIndices.Add(i);

            
            if(columnIndices.Count != 0 && rowIndices.Count != 0)
            {
                NDArray slicedArray = np.zeros((rowIndices.Count, columnIndices.Count, 2));
                
                for(int i = 0 ; i < rowIndices.Count; i++)
                {
                    int row_index = rowIndices[i];
                    int row = i;
                    for(int j = 0; j < columnIndices.Count; j++)
                    {
                    int col_index = columnIndices[j]; 
                    int col = j;
                    
                    var point1 = arr[row_index][col_index][0];
                    var point2 = arr[row_index][col_index][1];
                    
                    slicedArray[row][col][0] = point1;
                    slicedArray[row][col][1] = point2;
                    }
                }
                
                return slicedArray;
            }
            else
            {
                NDArray emptyArray = np.empty(rowIndices.Count,columnIndices.Count , 2);
                return emptyArray;            
            
            }
        }
    }
}