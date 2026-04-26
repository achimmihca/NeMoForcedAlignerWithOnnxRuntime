using System;
using System.IO;
using System.Linq;
using NUnit.Framework;
using Microsoft.ML.OnnxRuntime;
using NAudio;
using NWaves;

namespace NemoForcedAlignerWithOnnxRuntime
{
    [TestFixture]
    public class Tests
    {
        [Test]
        public void TestForcedAlignment()
        {
            // Find project root
            string currentDir = AppDomain.CurrentDomain.BaseDirectory;
            while (currentDir != null && !File.Exists(Path.Combine(currentDir, "NemoForcedAlignerWithOnnxRuntime.sln")))
            {
                currentDir = Path.GetDirectoryName(currentDir);
            }
            
            string projectRoot = currentDir ?? throw new Exception("Could not find project root");
            
            string modelPath = Path.Combine(projectRoot, "AiModels", "nfa_model.onnx");
            string tokensPath = Path.Combine(projectRoot, "AiModels", "tokens.txt");
            string audioPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav");
            string transcriptPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.txt");

            Assert.IsTrue(File.Exists(modelPath), $"Model not found at {modelPath}");

            var aligner = new NemoForcedAligner(modelPath, tokensPath);
            
            var samples = aligner.LoadAudio(audioPath);
            var features = aligner.ExtractFeatures(samples);
            var logprobs = aligner.RunInference(features);
            
            string transcript = File.ReadAllText(transcriptPath).Trim();
            var targetIds = aligner.Tokenize(transcript);
            
            var path = aligner.ViterbiAlign(logprobs, targetIds);
            var wordTimestamps = aligner.GetWordTimestamps(path, targetIds);

            foreach (var wt in wordTimestamps)
            {
                Console.WriteLine($"[DEBUG_LOG] {wt}");
            }

            Assert.IsNotEmpty(wordTimestamps);
            
            // The transcript is "I have that curiosity beside me at this moment" -> 9 words
            Assert.AreEqual(9, wordTimestamps.Count, "Should have 9 words");

            double lastEnd = 0;
            foreach (var wt in wordTimestamps)
            {
                Assert.GreaterOrEqual(wt.StartTime, 0, $"Start time for {wt.Word} should be >= 0");
                Assert.Greater(wt.EndTime, wt.StartTime, $"End time for {wt.Word} should be > Start time");
                Assert.GreaterOrEqual(wt.StartTime, lastEnd, $"Start time for {wt.Word} should be >= previous end time");
                lastEnd = wt.EndTime;
            }
            
            double audioDuration = samples.Length / 16000.0;
            Assert.LessOrEqual(wordTimestamps.Last().EndTime, audioDuration + 0.1, "End time should be within audio duration");
        }
    }
}
