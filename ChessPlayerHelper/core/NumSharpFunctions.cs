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
    
    }
}