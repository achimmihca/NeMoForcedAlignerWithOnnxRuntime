using System;
using System.Collections.Generic;
using NAudio.Wave;
using NAudio.Vorbis;

namespace NemoForcedAlignerWithOnnxRuntime
{
    public static class AudioLoader
    {
        public static AudioData LoadAudio(string path)
        {
            WaveStream reader;
            if (path.EndsWith(".ogg", StringComparison.OrdinalIgnoreCase))
            {
                reader = new VorbisWaveReader(path);
            }
            else
            {
                reader = new AudioFileReader(path);
            }

            using (reader)
            {
                var waveFormat = reader.WaveFormat;
                var sampleProvider = reader.ToSampleProvider();
                
                var samples = new List<float>();
                float[] buffer = new float[waveFormat.SampleRate];
                int read;
                while ((read = sampleProvider.Read(buffer, 0, buffer.Length)) > 0)
                {
                    for (int i = 0; i < read; i++) samples.Add(buffer[i]);
                }

                return new AudioData
                {
                    Samples = samples.ToArray(),
                    ChannelCount = waveFormat.Channels,
                    SampleRate = waveFormat.SampleRate
                };
            }
        }
    }
}
