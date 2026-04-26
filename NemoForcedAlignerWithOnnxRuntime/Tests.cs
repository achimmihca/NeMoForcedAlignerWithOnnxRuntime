using System;
using System.IO;
using System.Linq;
using NUnit.Framework;

namespace NemoForcedAlignerWithOnnxRuntime
{
    [TestFixture]
    public class Tests
    {
        [TestCase("en", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav", "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.txt", 9)]
        [TestCase("es", "Excerpt-Kurzgesagt-ComoFuncionaDeVerdadElSistemaInmunitario.ogg", "Excerpt-Kurzgesagt-ComoFuncionaDeVerdadElSistemaInmunitario.txt", 15)]
        [TestCase("de", "Excerpt-Kurzgesagt-DasImmunsystemErklärt.ogg", "Excerpt-Kurzgesagt-DasImmunsystemErklärt.txt", 14)]
        [TestCase("en", "Excerpt-Kurzgesagt-HowTheImmuneSystemActuallyWorks.ogg", "Excerpt-Kurzgesagt-HowTheImmuneSystemActuallyWorks.txt", 16)]
        public void TestForcedAlignment(string language, string audioFileName, string transcriptFileName, int expectedWordCount)
        {
            // Find project root
            string currentDir = AppDomain.CurrentDomain.BaseDirectory;
            while (currentDir != null && !File.Exists(Path.Combine(currentDir, "NemoForcedAlignerWithOnnxRuntime.sln")))
            {
                currentDir = Path.GetDirectoryName(currentDir);
            }
            
            string projectRoot = currentDir ?? throw new Exception("Could not find project root");

            var configs = new[]
            {
                new NemoForcedAlignerConfiguration("en", 
                    Path.Combine(projectRoot, "onnx_model_export", "stt_en_conformer_ctc_small.onnx"),
                    Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_en_conformer_ctc_small.txt")),
                new NemoForcedAlignerConfiguration("de", 
                    Path.Combine(projectRoot, "onnx_model_export", "stt_de_conformer_ctc_large.onnx"),
                    Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_de_conformer_ctc_large.txt")),
                new NemoForcedAlignerConfiguration("es", 
                    Path.Combine(projectRoot, "onnx_model_export", "stt_es_conformer_ctc_large.onnx"),
                    Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_es_conformer_ctc_large.txt"))
            };

            var config = configs.FirstOrDefault(c => c.Language == language) 
                         ?? throw new Exception($"No configuration found for language {language}");
            
            string audioPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", audioFileName);
            string transcriptPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", transcriptFileName);

            Assert.IsTrue(File.Exists(config.ModelPath), $"Model not found at {config.ModelPath}");

            var aligner = new NemoForcedAligner(config);
            
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
