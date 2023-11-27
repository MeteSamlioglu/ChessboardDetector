/* 
    A module that combines chess board detection modules in this files
*/
using NumSharp;

class Program
{
    static void Main()
    {
        // Create a 1D array
        var array1D = np.arange(0, 10);
        Console.WriteLine("1D Array: " + array1D);

        // Create a 2D array
        var array2D = np.arange(0, 12).reshape(3, 4);
        Console.WriteLine("2D Array:");
        //Accessing elements
        Console.WriteLine("Element at (1, 2): " + array2D[1, 2]);

        //Perform operations
        var sum = np.sum(array2D);
        Console.WriteLine("Sum of array2D: " + sum);
    }
}