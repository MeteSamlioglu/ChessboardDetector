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
        
        public static NDArray SliceIntegerNDArray(NDArray arr, NDArray rows_to_slice, NDArray cols_to_slice )
        {
            
            List<int> rowIndices = new List<int>();
            List<int> columnIndices = new List<int>();
            
            for(int i = 0 ; i < rows_to_slice.shape[0]; i++)
                     rowIndices.Add(i);
            
            for(int i = 0 ; i < cols_to_slice.shape[0]; i++)
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
    
        public static ValueTuple<NDArray, NDArray, NDArray, NDArray> Unique(NDArray col_xs, NDArray row_ys)
        {
                 
            int[] col_xs_ = col_xs.ToArray<int>();
            int[] row_ys_ = row_ys.ToArray<int>();

            var colXsWithIndices = col_xs_.Select((value, index) => new { Value = value, Index = index })
                                        .GroupBy(item => item.Value)
                                        .Select(group => group.First());

            var rowYsWithIndices = row_ys_.Select((value, index) => new { Value = value, Index = index })
                                        .GroupBy(item => item.Value)
                                        .Select(group => group.First());

            int[] uniqueColXs = colXsWithIndices.Select(item => item.Value).ToArray();
            int[] colIndices = colXsWithIndices.Select(item => item.Index).ToArray();

            int[] uniqueRowYs = rowYsWithIndices.Select(item => item.Value).ToArray();
            int[] rowIndices = rowYsWithIndices.Select(item => item.Index).ToArray();
            
            NDArray _col_xs = np.array(uniqueColXs);
            
            NDArray _row_ys = np.array(uniqueRowYs);
            
            NDArray col_indices = np.array(colIndices);
            
            NDArray row_indices = np.array(rowIndices);

            return new ValueTuple<NDArray, NDArray, NDArray, NDArray>(_col_xs, col_indices, _row_ys, row_indices); 
        }
        
        public static NDArray Mask(NDArray arr, int min, int max)
        {
            List<bool> maskMin = new List<bool>();
            
            List<bool> maskMax = new List<bool>();
            
            for(int i = 0 ; i < arr.shape[0]; i++)
            {
                int var = arr[i];
                                
                if(var >= min)
                    maskMin.Add(true);
                
                else
                    maskMin.Add(false);
            }

            for(int i = 0 ; i < arr.shape[0]; i++)
            {
                int var = arr[i];
                
                if(var <= max)
                    maskMax.Add(true);
                
                else
                    maskMax.Add(false);
            }
            
            NDArray mask1 = np.array(maskMin);
            
            NDArray mask2 = np.array(maskMax);
            
            var col_mask = np.logical_and(mask1, mask2);
            
            return col_mask;
        }

        public static NDArray Slice1DBoolean(NDArray arr, NDArray bool_arr)
        {
            
            
                List<int> elements_to_consider = new List<int>();
                
                for(int i = 0 ; i < bool_arr.shape[0]; i++)
                {
                    if(bool_arr[i])
                        elements_to_consider.Add(arr[i]);
                }

                if(elements_to_consider.Count > 0)
                {
                    NDArray slicedArr = np.array(elements_to_consider);
                    
                    return slicedArr;
                }
                else
                {
          
                    NDArray emptyArray = np.empty(elements_to_consider.Count,1);
                    return emptyArray;
                }
            
        } 


        public static NDArray MultNDArray(NDArray arr, int horizontal_scale, int vertical_scale)
        {
            
            for(int i = 0 ; i < arr.shape[0]; i++)
            {
                for(int j = 0 ; j < arr.shape[1]; j++)
                {
                    var element1 = arr[i][j][0] * horizontal_scale;
                    var element2 = arr[i][j][1] * vertical_scale;

                    arr[i][j][0] = element1;
                    arr[i][j][1] = element2;
                }
            }
            return arr;
        }
    }
}