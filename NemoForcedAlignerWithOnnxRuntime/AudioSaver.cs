using System.IO;
using NAudio.Wave;

namespace NemoForcedAlignerWithOnnxRuntime
{
    public static class AudioSaver
    {
        public static void SaveAudio(string path, float[] samples, int sampleRate, int channelCount)
        {
            var folder = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(folder) && !Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
            }

            using (var writer = new WaveFileWriter(path, WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channelCount)))
            {
                writer.WriteSamples(samples, 0, samples.Length);
            }
        }
    }
}
