namespace Wikiled.MachineLearning.Svm.Logic
{
    /// <summary>
    /// Class encoding the point on a 2D curve.
    /// </summary>
    public class CurvePoint
    {
        /// <summary>
        /// Constructor.
        /// </summary>
        /// <param name="x">X coordinate</param>
        /// <param name="y">Y coordinate</param>
        public CurvePoint(float x, float y)
        {
            X = x;
            Y = y;
        }

        /// <summary>
        /// X coordinate
        /// </summary>
        public float X { get; }

        /// <summary>
        /// Y coordinate
        /// </summary>
        public float Y { get; }

        /// <summary>
        /// Creates a string representation of this point.
        /// </summary>
        /// <returns>string in the form (x, y)</returns>
        public override string ToString()
        {
            return $"({X}, {Y})";
        }
    }
}