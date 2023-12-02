using System.Collections.Generic;

namespace Data
{
    public static class CONFIGURATION
    {
        public static class RESIZE_IMAGE
        {   
            public static readonly int WIDTH = 1200;
        }

        public static class EDGE_DETECTION
        {   
            private static readonly int APERTURE = 3;
            private static readonly int HIGH_THRESHOLD = 400;

            private static readonly int LOW_THRESHOLD = 90;
        }

        public static class LINE_DETECTION
        {
            private static readonly int THRESHOLD = 150;
            private static readonly bool DIAGONAL_LINE_ELIMINATION = true;

            private static readonly int  DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES =  30;
        }

        public static class BORDER_REFINEMENT
        {
            public static readonly int LINE_WIDTH = 4;
            public static readonly List<int> WARPED_SQUARE_SIZE = new List<int> { 50, 50 };

            public static readonly int NUM_SURROUNDING_SQUARES_IN_WARPED_IMG = 5;

            public static readonly int SOBEL_KERNEL_SIZE = 3;
            
            public static class EDGE_DETECTION
            {  
                public static class HORIZONTAL
                {
                    public static readonly int APERTURE = 3;
                    public static readonly int HIGH_THRESHOLD = 300;
                    public static readonly int LOW_THRESHOLD = 120;
                }
                public static class VERTICAL
                {
                    public static readonly int APERTURE = 3;
                    public static readonly int HIGH_THRESHOLD = 200;
                    public static readonly int LOW_THRESHOLD = 100;
                }
            }
        }

        public static readonly float MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE = 0.7F;

        public static class RANSAC
        {
            public static readonly float BEST_SOLUTION_TOLERANCE = 0.15F;
            public static readonly float OFFSET_TOLERANCE = 0.1F;

        }
    }    
}