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
        [TestCase("en", "Excerpt-Kurzgesagt-HowTheImmuneSystemActuallyWorks.ogg", "Excerpt-Kurzgesagt-HowTheImmuneSystemActuallyWorks.txt", 16)]
        [TestCase("de", "Excerpt-Kurzgesagt-DasImmunsystemErklärt.ogg", "Excerpt-Kurzgesagt-DasImmunsystemErklärt.txt", 14)]
        [TestCase("es", "Excerpt-Kurzgesagt-ComoFuncionaDeVerdadElSistemaInmunitario.ogg", "Excerpt-Kurzgesagt-ComoFuncionaDeVerdadElSistemaInmunitario.txt", 15)]
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
                new NemoForcedAligner.Configuration("en", 
                    Path.Combine(projectRoot, "onnx_model_export", "stt_en_conformer_ctc_large.onnx"),
                    Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_en_conformer_ctc_large.txt")),
                new NemoForcedAligner.Configuration("de", 
                    Path.Combine(projectRoot, "onnx_model_export", "stt_de_conformer_ctc_large.onnx"),
                    Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_de_conformer_ctc_large.txt")),
                new NemoForcedAligner.Configuration("es", 
                    Path.Combine(projectRoot, "onnx_model_export", "stt_es_conformer_ctc_large.onnx"),
                    Path.Combine(projectRoot, "onnx_model_export", "tokens_stt_es_conformer_ctc_large.txt"))
            };

            var config = configs.FirstOrDefault(c => c.Language == language) 
                         ?? throw new Exception($"No configuration found for language {language}");
            
            string audioPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", audioFileName);
            string transcriptPath = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestData", transcriptFileName);
            string transcript = File.ReadAllText(transcriptPath).Replace("\n", " ").Trim();

            Assert.IsTrue(File.Exists(config.ModelPath), $"Model not found at {config.ModelPath}");

            var audioData = AudioLoader.LoadAudio(audioPath);
            NemoForcedAligner.ForcedAlignmentResult alignment;
            using (var aligner = new NemoForcedAligner(config.ModelPath, config.TokensPath))
            {
                alignment = aligner.Run(audioData, transcript);
            }

            foreach (var wordTimestamp in alignment.Words)
            {
                Console.WriteLine($"{wordTimestamp}");
            }

            Assert.IsNotEmpty(alignment.Words);
            Assert.IsNotEmpty(alignment.Tokens);

            if (expectedWordCount > 0)
            {
                Assert.AreEqual(expectedWordCount, alignment.Words.Count, $"Should have {expectedWordCount} words");
            }

            double lastEnd = 0;
            foreach (var wt in alignment.Words)
            {
                Assert.GreaterOrEqual(wt.StartTime, 0, $"Start time for {wt.Word} should be >= 0");
                Assert.Greater(wt.EndTime, wt.StartTime, $"End time for {wt.Word} should be > Start time");
                Assert.GreaterOrEqual(wt.StartTime, lastEnd, $"Start time for {wt.Word} should be >= previous end time");
                lastEnd = wt.EndTime;

                double lastTokenEnd = wt.StartTime;
                foreach (var tt in wt.Tokens)
                {
                    Assert.GreaterOrEqual(tt.StartTime, lastTokenEnd, $"Token {tt.Token} start time should be >= previous token end time");
                    Assert.Greater(tt.EndTime, tt.StartTime, $"Token {tt.Token} end time should be > start time");
                    Assert.LessOrEqual(tt.EndTime, wt.EndTime + 0.001, $"Token {tt.Token} end time should be <= word end time");
                    lastTokenEnd = tt.EndTime;
                }
            }
            
            double audioDuration = audioData.Samples.Length / (double)audioData.SampleRate;
            Assert.LessOrEqual(alignment.Words.Last().EndTime, audioDuration + 0.1, "End time should be within audio duration");

            // Extract audio snippets for words
            string testResultsBase = Path.Combine(projectRoot, "NemoForcedAlignerWithOnnxRuntime", "TestResults");
            string audioOutputFolder = Path.Combine(testResultsBase, Path.GetFileNameWithoutExtension(audioFileName));
            if (Directory.Exists(audioOutputFolder))
            {
                Directory.Delete(audioOutputFolder, true);
            }
            Directory.CreateDirectory(audioOutputFolder);

            for (int i = 0; i < alignment.Words.Count; i++)
            {
                var wordTimestamp = alignment.Words[i];
                int startFrame = (int)(wordTimestamp.StartTime * audioData.SampleRate);
                int endFrame = (int)(wordTimestamp.EndTime * audioData.SampleRate);
                
                // Ensure frames are within range
                int totalFrames = audioData.Samples.Length / audioData.ChannelCount;
                startFrame = Math.Max(0, Math.Min(startFrame, totalFrames));
                endFrame = Math.Max(0, Math.Min(endFrame, totalFrames));
                
                if (endFrame > startFrame)
                {
                    int startSample = startFrame * audioData.ChannelCount;
                    int endSample = endFrame * audioData.ChannelCount;
                    float[] wordSamples = new float[endSample - startSample];
                    Array.Copy(audioData.Samples, startSample, wordSamples, 0, wordSamples.Length);
                    
                    string safeWord = string.Concat(wordTimestamp.Word.Where(c => !Path.GetInvalidFileNameChars().Contains(c)));
                    string wordAudioFileName = $"{(i + 1):D2} - {safeWord}.wav";
                    string wordAudioPath = Path.Combine(audioOutputFolder, wordAudioFileName);
                    AudioSaver.SaveAudio(wordAudioPath, wordSamples, audioData.SampleRate, audioData.ChannelCount);
                }
            }
        }
    }
}
