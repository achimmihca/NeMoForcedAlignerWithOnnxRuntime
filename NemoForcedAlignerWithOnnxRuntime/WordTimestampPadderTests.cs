using System.Collections.Generic;
using NUnit.Framework;

namespace NemoForcedAlignerWithOnnxRuntime
{
    [TestFixture]
    public class WordTimestampPadderTests
    {
        [Test]
        public void TestPadding_NoOverlap()
        {
            var alignment = new NemoForcedAligner.ForcedAlignmentResult
            {
                Words = new List<NemoForcedAligner.WordTimestamp>
                {
                    new NemoForcedAligner.WordTimestamp { Word = "one", StartTime = 1.0, EndTime = 2.0 },
                    new NemoForcedAligner.WordTimestamp { Word = "two", StartTime = 3.0, EndTime = 4.0 }
                }
            };

            var padder = new WordTimestampPadder(200, 200);
            var result = padder.PadTimestamps(alignment);

            Assert.AreEqual(0.8, result.Words[0].StartTime, 1e-6);
            Assert.AreEqual(2.2, result.Words[0].EndTime, 1e-6);
            Assert.AreEqual(2.8, result.Words[1].StartTime, 1e-6);
            Assert.AreEqual(4.2, result.Words[1].EndTime, 1e-6);

            // Verify original not modified
            Assert.AreEqual(1.0, alignment.Words[0].StartTime, 1e-6);
        }

        [Test]
        public void TestPadding_WithOverlapMitigation()
        {
            var alignment = new NemoForcedAligner.ForcedAlignmentResult
            {
                Words = new List<NemoForcedAligner.WordTimestamp>
                {
                    new NemoForcedAligner.WordTimestamp { Word = "one", StartTime = 1.0, EndTime = 2.0 },
                    new NemoForcedAligner.WordTimestamp { Word = "two", StartTime = 2.1, EndTime = 3.1 }
                }
            };

            // Gap is 0.1s. Padding requested is 0.2s end for "one" and 0.2s start for "two". Total 0.4s.
            // Proportional distribution should give each 0.05s.
            var padder = new WordTimestampPadder(200, 200);
            var result = padder.PadTimestamps(alignment);

            Assert.AreEqual(0.8, result.Words[0].StartTime, 1e-6);
            Assert.AreEqual(2.05, result.Words[0].EndTime, 1e-6);
            Assert.AreEqual(2.05, result.Words[1].StartTime, 1e-6);
            Assert.AreEqual(3.3, result.Words[1].EndTime, 1e-6);
        }

        [Test]
        public void TestPadding_WordLengthFilter()
        {
            var alignment = new NemoForcedAligner.ForcedAlignmentResult
            {
                Words = new List<NemoForcedAligner.WordTimestamp>
                {
                    new NemoForcedAligner.WordTimestamp { Word = "short", StartTime = 1.0, EndTime = 1.1 }, // 100ms
                    new NemoForcedAligner.WordTimestamp { Word = "long", StartTime = 2.0, EndTime = 2.3 }   // 300ms
                }
            };

            var padder = new WordTimestampPadder(50, 50, 200); // Only pad words < 200ms
            var result = padder.PadTimestamps(alignment);

            Assert.AreEqual(0.95, result.Words[0].StartTime, 1e-6);
            Assert.AreEqual(1.15, result.Words[0].EndTime, 1e-6);
            Assert.AreEqual(2.0, result.Words[1].StartTime, 1e-6);
            Assert.AreEqual(2.3, result.Words[1].EndTime, 1e-6);
        }

        [Test]
        public void TestPadding_DefaultZero()
        {
            var alignment = new NemoForcedAligner.ForcedAlignmentResult
            {
                Words = new List<NemoForcedAligner.WordTimestamp>
                {
                    new NemoForcedAligner.WordTimestamp { Word = "one", StartTime = 1.0, EndTime = 2.0 }
                }
            };

            var padder = new WordTimestampPadder();
            var result = padder.PadTimestamps(alignment);

            Assert.AreEqual(1.0, result.Words[0].StartTime, 1e-6);
            Assert.AreEqual(2.0, result.Words[0].EndTime, 1e-6);
        }
        
        [Test]
        public void TestPadding_TokensUpdated()
        {
            var tt1 = new NemoForcedAligner.TokenTimestamp { Token = "o", StartTime = 1.0, EndTime = 1.5 };
            var tt2 = new NemoForcedAligner.TokenTimestamp { Token = "ne", StartTime = 1.5, EndTime = 2.0 };
            var wt = new NemoForcedAligner.WordTimestamp { Word = "one", StartTime = 1.0, EndTime = 2.0 };
            wt.Tokens.Add(tt1);
            wt.Tokens.Add(tt2);
            
            var alignment = new NemoForcedAligner.ForcedAlignmentResult
            {
                Words = new List<NemoForcedAligner.WordTimestamp> { wt }
            };

            var padder = new WordTimestampPadder(100, 100);
            var result = padder.PadTimestamps(alignment);

            Assert.AreEqual(0.9, result.Words[0].StartTime, 1e-6);
            Assert.AreEqual(2.1, result.Words[0].EndTime, 1e-6);
            
            var resT1 = result.Words[0].Tokens[0];
            var resT2 = result.Words[0].Tokens[1];

            Assert.AreEqual(0.9, resT1.StartTime, 1e-6);
            Assert.AreEqual(1.5, resT1.EndTime, 1e-6);
            Assert.AreEqual(1.5, resT2.StartTime, 1e-6);
            Assert.AreEqual(2.1, resT2.EndTime, 1e-6);
            
            // Verify original tokens NOT updated
            Assert.AreEqual(1.0, tt1.StartTime, 1e-6);
            Assert.AreEqual(2.0, tt2.EndTime, 1e-6);
        }

        [Test]
        public void TestPadding_MaxEndTime()
        {
            var alignment = new NemoForcedAligner.ForcedAlignmentResult
            {
                Words = new List<NemoForcedAligner.WordTimestamp>
                {
                    new NemoForcedAligner.WordTimestamp { Word = "one", StartTime = 1.0, EndTime = 2.0 }
                }
            };

            // Max end time is 2.1s. Padding requested is 200ms (to 2.2s). 
            // It should be capped at 2.1s.
            var padder = new WordTimestampPadder(0, 200, null, 2100);
            var result = padder.PadTimestamps(alignment);

            Assert.AreEqual(2.1, result.Words[0].EndTime, 1e-6);
        }
    }
}
