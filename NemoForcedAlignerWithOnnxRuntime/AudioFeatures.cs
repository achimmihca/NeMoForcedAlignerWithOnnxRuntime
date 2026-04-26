namespace NemoForcedAlignerWithOnnxRuntime
{
    public class AudioFeatures
    {
        public float[][] Data { get; set; }
        
        public int FrameCount => Data?.Length ?? 0;
        public int FeatureCount => (Data != null && Data.Length > 0) ? Data[0].Length : 0;
    }
}
