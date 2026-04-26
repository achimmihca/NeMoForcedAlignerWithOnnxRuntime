using System;
using System.IO;
using System.Linq;
using NUnit.Framework;

namespace NemoForcedAlignerWithOnnxRuntime
{
    [TestFixture]
    public class Tests
    {
        [TestCase("Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.txt", 9)]
        [TestCase("Excerpt-Kurzgesagt-ComoFuncionaDeVerdadElSistemaInmunitario.ogg", "Excerpt-Kurzgesagt-ComoFuncionaDeVerdadElSistemaInmunitario.txt", 15)]
        [TestCase("Excerpt-Kurzgesagt-DasImmunsystemErklärt.ogg", "Excerpt-Kurzgesagt-DasImmunsystemErklärt.txt", 14)]
        [TestCase("Excerpt-Kurzgesagt-HowTheImmuneSystemActuallyWorks.ogg", "Excerpt-Kurzgesagt-HowTheImmuneSystemActuallyWorks.txt", 16)]
        public void TestForcedAlignment(string audioFileName, string transcriptFileName, int expectedWordCount)
        {
            // Find project root
            string currentDir = AppDomain.CurrentDomain.BaseDirectory;
            while (currentDir != null && !File.Exists(Path.Combine(currentDir, "NemoForcedAlignerWithOnnxRuntime.sln")))
            {
                currentDir = Path.GetDirectoryName(currentDir);
            }
            
            string projectRoot = currentDir ?? throw new Exception("Could not find project root");
            
            string modelPath = Path.Combine(projectRoot, "onnx_model_export", "stt_en_conformer_ctc_small.onnx");
            string tokensPath = Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_en_conformer_ctc_small.txt");
            string audioPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", audioFileName);
            string transcriptPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", transcriptFileName);

            Assert.IsTrue(File.Exists(modelPath), $"Model not found at {modelPath}");

            var aligner = new NemoForcedAligner(modelPath, tokensPath);
            
            var audioData = AudioLoader.LoadAudio(audioPath);
            var features = aligner.ExtractFeatures(audioData);
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
            
            Assert.AreEqual(expectedWordCount, wordTimestamps.Count, $"Should have {expectedWordCount} words");

            double lastEnd = 0;
            foreach (var wt in wordTimestamps)
            {
                Assert.GreaterOrEqual(wt.StartTime, 0, $"Start time for {wt.Word} should be >= 0");
                Assert.Greater(wt.EndTime, wt.StartTime, $"End time for {wt.Word} should be > Start time");
                Assert.GreaterOrEqual(wt.StartTime, lastEnd, $"Start time for {wt.Word} should be >= previous end time");
                lastEnd = wt.EndTime;
            }
            
            double audioDuration = audioData.Samples.Length / (double)audioData.SampleRate;
            Assert.LessOrEqual(wordTimestamps.Last().EndTime, audioDuration + 0.1, "End time should be within audio duration");
        }
    }
}
