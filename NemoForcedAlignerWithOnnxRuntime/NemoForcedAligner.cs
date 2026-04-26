using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;
using NWaves.Windows;
using NWaves.Filters.Fda;
using NWaves.Signals;
using NWaves.Operations;

namespace NemoForcedAlignerWithOnnxRuntime
{
    public class NemoForcedAligner
    {
        private readonly string[] _tokens;
        private readonly Dictionary<string, int> _tokenToId;
        private readonly InferenceSession _session;
        private const int BlankIndex = 1024;
        private const int SampleRate = 16000;
        private const int DownsamplingFactor = 4;

        public NemoForcedAligner(string modelPath, string tokensPath)
        {
            _session = new InferenceSession(modelPath);
            _tokens = File.ReadAllLines(tokensPath).Where(l => !string.IsNullOrWhiteSpace(l)).ToArray();
            _tokenToId = new Dictionary<string, int>();
            for (int i = 0; i < _tokens.Length; i++)
            {
                _tokenToId[_tokens[i]] = i;
            }
        }

        public List<int> Tokenize(string text)
        {
            text = text.ToLowerInvariant().Replace(" ", "▁");
            if (!text.StartsWith("▁")) text = "▁" + text;

            var result = new List<int>();
            int pos = 0;
            while (pos < text.Length)
            {
                bool found = false;
                for (int len = Math.Min(text.Length - pos, 20); len > 0; len--)
                {
                    string sub = text.Substring(pos, len);
                    if (_tokenToId.TryGetValue(sub, out int id))
                    {
                        result.Add(id);
                        pos += len;
                        found = true;
                        break;
                    }
                }
                if (!found) pos++;
            }
            return result;
        }


        public AudioFeatures ExtractFeatures(AudioData audioData)
        {
            float[] samples = audioData.Samples;
            if (audioData.SampleRate != SampleRate || audioData.ChannelCount != 1)
            {
                // Convert to mono if needed
                if (audioData.ChannelCount > 1)
                {
                    float[] monoSamples = new float[audioData.Samples.Length / audioData.ChannelCount];
                    for (int i = 0; i < monoSamples.Length; i++)
                    {
                        float sum = 0;
                        for (int c = 0; c < audioData.ChannelCount; c++)
                        {
                            sum += audioData.Samples[i * audioData.ChannelCount + c];
                        }
                        monoSamples[i] = sum / audioData.ChannelCount;
                    }
                    samples = monoSamples;
                }

                // Resample if needed
                if (audioData.SampleRate != SampleRate)
                {
                    var inputSignal = new DiscreteSignal(audioData.SampleRate, samples);
                    var resampled = Operation.Resample(inputSignal, SampleRate);
                    samples = resampled.Samples;
                }
            }

            var options = new FilterbankOptions
            {
                SamplingRate = SampleRate,
                FrameSize = 400,
                HopSize = 160,
                Window = WindowType.Hann,
                FilterBankSize = 80,
                LowFrequency = 0,
                HighFrequency = 8000,
                FftSize = 512,
                FilterBank = FilterBanks.Triangular(512, SampleRate, FilterBanks.MelBands(80, SampleRate, 0, 8000))
            };
            
            var extractor = new FilterbankExtractor(options);
            var signal = new DiscreteSignal(SampleRate, samples);
            var features = extractor.ComputeFrom(signal);
            
            foreach (var frame in features)
            {
                for (int i = 0; i < frame.Length; i++)
                {
                    frame[i] = (float)Math.Log(Math.Max(frame[i], 1e-5));
                }
            }
            
            return new AudioFeatures { Data = features.ToArray() };
        }


        public LogProbs RunInference(AudioFeatures features)
        {
            int numFrames = features.FrameCount;
            int numBins = features.FeatureCount;
            
            var tensor = new DenseTensor<float>(new[] { 1, numBins, numFrames });
            for (int t = 0; t < numFrames; t++)
            {
                for (int b = 0; b < numBins; b++)
                {
                    tensor[0, b, t] = features.Data[t][b];
                }
            }
            
            var lengthTensor = new DenseTensor<long>(new[] { 1 });
            lengthTensor[0] = numFrames;
            
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("audio_signal", tensor),
                NamedOnnxValue.CreateFromTensor("length", lengthTensor)
            };
            
            using (var results = _session.Run(inputs))
            {
                var logprobsTensor = results.First(r => r.Name == "logprobs").AsTensor<float>();
                int outFrames = logprobsTensor.Dimensions[1];
                int vocabSize = logprobsTensor.Dimensions[2];
                
                var output = new float[outFrames, vocabSize];
                for (int t = 0; t < outFrames; t++)
                {
                    for (int v = 0; v < vocabSize; v++)
                    {
                        output[t, v] = logprobsTensor[0, t, v];
                    }
                }
                return new LogProbs { Data = output };
            }
        }

        public int[] ViterbiAlign(LogProbs logProbs, List<int> targetIds)
        {
            float[,] logprobs = logProbs.Data;
            int T = logprobs.GetLength(0);
            int[] augmented = new int[2 * targetIds.Count + 1];
            for (int i = 0; i < targetIds.Count; i++)
            {
                augmented[2 * i] = BlankIndex;
                augmented[2 * i + 1] = targetIds[i];
            }
            augmented[augmented.Length - 1] = BlankIndex;

            int S = augmented.Length;
            float[,] dp = new float[T, S];
            int[,] backtrack = new int[T, S];

            for (int t = 0; t < T; t++)
                for (int s = 0; s < S; s++)
                    dp[t, s] = float.NegativeInfinity;

            dp[0, 0] = logprobs[0, augmented[0]];
            dp[0, 1] = logprobs[0, augmented[1]];

            for (int t = 1; t < T; t++)
            {
                for (int s = 0; s < S; s++)
                {
                    // Stay in same state
                    float bestPrevLogProb = dp[t - 1, s];
                    int bestPrevState = s;

                    // From s-1
                    if (s > 0 && dp[t - 1, s - 1] > bestPrevLogProb)
                    {
                        bestPrevLogProb = dp[t - 1, s - 1];
                        bestPrevState = s - 1;
                    }

                    // From s-2 (if current is not blank and s-1 is blank and target[s] != target[s-2])
                    if (s > 1 && augmented[s] != BlankIndex && augmented[s] != augmented[s - 2])
                    {
                        if (dp[t - 1, s - 2] > bestPrevLogProb)
                        {
                            bestPrevLogProb = dp[t - 1, s - 2];
                            bestPrevState = s - 2;
                        }
                    }

                    if (!float.IsNegativeInfinity(bestPrevLogProb))
                    {
                        dp[t, s] = bestPrevLogProb + logprobs[t, augmented[s]];
                        backtrack[t, s] = bestPrevState;
                    }
                }
            }

            int[] path = new int[T];
            int currentS = (dp[T - 1, S - 1] > dp[T - 1, S - 2]) ? S - 1 : S - 2;
            for (int t = T - 1; t >= 0; t--)
            {
                path[t] = augmented[currentS];
                currentS = backtrack[t, currentS];
            }

            return path;
        }

        public List<WordTimestamp> GetWordTimestamps(int[] path, List<int> targetIds)
        {
            var result = new List<WordTimestamp>();
            int targetIdx = 0;
            
            double frameDuration = 0.01 * DownsamplingFactor; // 10ms hop * downsampling

            int currentWordStartFrame = -1;
            string currentWord = "";

            for (int t = 0; t < path.Length; t++)
            {
                int tokenId = path[t];
                if (tokenId != BlankIndex)
                {
                    if (targetIdx < targetIds.Count && tokenId == targetIds[targetIdx])
                    {
                        string token = _tokens[tokenId];
                        if (token.StartsWith("▁"))
                        {
                            // New word starts
                            if (!string.IsNullOrEmpty(currentWord))
                            {
                                // End previous word
                                // We'll estimate end time as current frame
                            }
                            // Actually we need to be more careful. 
                            // A word starts when we first see its first token.
                        }
                        // This logic is a bit flawed because we might see the same token multiple times.
                        // But for forced alignment, we know the sequence of tokens.
                    }
                }
            }
            
            // Simpler approach: find the first frame for each token in targetIds
            int[] tokenStartFrames = new int[targetIds.Count];
            int currentT = 0;
            for (int i = 0; i < targetIds.Count; i++)
            {
                while (currentT < path.Length && path[currentT] != targetIds[i])
                {
                    currentT++;
                }
                tokenStartFrames[i] = currentT;
                // Move past this token in path to find next one
                while (currentT < path.Length && path[currentT] == targetIds[i])
                {
                    currentT++;
                }
            }

            for (int i = 0; i < targetIds.Count; i++)
            {
                string token = _tokens[targetIds[i]];
                if (token.StartsWith("▁"))
                {
                    var wt = new WordTimestamp
                    {
                        Word = token.Substring(1),
                        StartTime = tokenStartFrames[i] * frameDuration
                    };
                    result.Add(wt);
                }
                else if (result.Count > 0)
                {
                    result.Last().Word += token;
                }
            }
            
            // Set end times
            for (int i = 0; i < result.Count; i++)
            {
                if (i < result.Count - 1)
                    result[i].EndTime = result[i + 1].StartTime;
                else
                    result[i].EndTime = path.Length * frameDuration;
            }

            return result;
        }
    }

    public class WordTimestamp
    {
        public string Word { get; set; }
        public double StartTime { get; set; }
        public double EndTime { get; set; }
        public override string ToString() => $"{Word}: {StartTime:F2} - {EndTime:F2}";
    }
}
