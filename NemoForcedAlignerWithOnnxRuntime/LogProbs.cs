namespace NemoForcedAlignerWithOnnxRuntime
{
    public class LogProbs
    {
        public float[,] Data { get; set; }
        
        public int FrameCount => Data?.GetLength(0) ?? 0;
        public int VocabSize => Data?.GetLength(1) ?? 0;
    }
}
