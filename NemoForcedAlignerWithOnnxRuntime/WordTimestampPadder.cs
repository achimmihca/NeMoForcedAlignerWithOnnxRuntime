using System;
using System.Collections.Generic;
using System.Linq;

namespace NemoForcedAlignerWithOnnxRuntime
{
    public class WordTimestampPadder
    {
        public double StartPaddingMs { get; set; }
        public double EndPaddingMs { get; set; }
        public double? WordLengthFilterMs { get; set; }
        public double? MaxEndTimeMs { get; set; }

        public WordTimestampPadder(double startPaddingMs = 0, double endPaddingMs = 0, double? wordLengthFilterMs = null, double? maxEndTimeMs = null)
        {
            StartPaddingMs = startPaddingMs;
            EndPaddingMs = endPaddingMs;
            WordLengthFilterMs = wordLengthFilterMs;
            MaxEndTimeMs = maxEndTimeMs;
        }

        public NemoForcedAligner.ForcedAlignmentResult PadTimestamps(NemoForcedAligner.ForcedAlignmentResult alignment)
        {
            if (alignment == null)
            {
                return null;
            }

            if (alignment.Words == null || alignment.Words.Count == 0 || (StartPaddingMs == 0 && EndPaddingMs == 0))
            {
                return CloneResult(alignment);
            }

            var clonedAlignment = CloneResult(alignment);
            var words = clonedAlignment.Words;

            double startPaddingSec = StartPaddingMs / 1000.0;
            double endPaddingSec = EndPaddingMs / 1000.0;

            int n = words.Count;
            var shouldPad = new bool[n];
            for (int i = 0; i < n; i++)
            {
                double durationMs = (words[i].EndTime - words[i].StartTime) * 1000.0;
                shouldPad[i] = !WordLengthFilterMs.HasValue || durationMs < WordLengthFilterMs.Value;
            }

            // Pad first word's start
            if (shouldPad[0])
            {
                double availableStart = words[0].StartTime;
                double padding = Math.Min(startPaddingSec, availableStart);
                words[0].StartTime -= padding;
                UpdateFirstTokenStart(words[0]);
            }

            // Pad gaps between words
            for (int i = 0; i < n - 1; i++)
            {
                double currentEnd = words[i].EndTime;
                double nextStart = words[i + 1].StartTime;
                double gap = Math.Max(0, nextStart - currentEnd);

                double requestedEndPadding = shouldPad[i] ? endPaddingSec : 0;
                double requestedStartPadding = shouldPad[i + 1] ? startPaddingSec : 0;

                double totalRequested = requestedEndPadding + requestedStartPadding;

                if (totalRequested > 0)
                {
                    double actualEndPadding;
                    double actualStartPadding;

                    if (totalRequested <= gap)
                    {
                        actualEndPadding = requestedEndPadding;
                        actualStartPadding = requestedStartPadding;
                    }
                    else
                    {
                        double factor = gap / totalRequested;
                        actualEndPadding = requestedEndPadding * factor;
                        actualStartPadding = requestedStartPadding * factor;
                    }

                    words[i].EndTime += actualEndPadding;
                    words[i + 1].StartTime -= actualStartPadding;
                    
                    UpdateLastTokenEnd(words[i]);
                    UpdateFirstTokenStart(words[i + 1]);
                }
            }

            // Pad last word's end
            if (shouldPad[n - 1])
            {
                double padding = endPaddingSec;
                if (MaxEndTimeMs.HasValue)
                {
                    double maxEndSec = MaxEndTimeMs.Value / 1000.0;
                    double available = Math.Max(0, maxEndSec - words[n - 1].EndTime);
                    padding = Math.Min(padding, available);
                }
                
                words[n - 1].EndTime += padding;
                UpdateLastTokenEnd(words[n - 1]);
            }

            return clonedAlignment;
        }

        private void UpdateFirstTokenStart(NemoForcedAligner.WordTimestamp word)
        {
            if (word.Tokens != null && word.Tokens.Count > 0)
            {
                word.Tokens[0].StartTime = word.StartTime;
            }
        }

        private void UpdateLastTokenEnd(NemoForcedAligner.WordTimestamp word)
        {
            if (word.Tokens != null && word.Tokens.Count > 0)
            {
                word.Tokens[word.Tokens.Count - 1].EndTime = word.EndTime;
            }
        }

        private static NemoForcedAligner.ForcedAlignmentResult CloneResult(NemoForcedAligner.ForcedAlignmentResult alignment)
        {
            var result = new NemoForcedAligner.ForcedAlignmentResult();
            var tokenMap = new Dictionary<NemoForcedAligner.TokenTimestamp, NemoForcedAligner.TokenTimestamp>();
            foreach (var t in alignment.Tokens)
            {
                var clonedToken = CloneToken(t);
                result.Tokens.Add(clonedToken);
                tokenMap[t] = clonedToken;
            }
            foreach (var w in alignment.Words)
            {
                result.Words.Add(CloneWord(w, tokenMap));
            }
            return result;
        }

        private static NemoForcedAligner.TokenTimestamp CloneToken(NemoForcedAligner.TokenTimestamp token)
        {
            return new NemoForcedAligner.TokenTimestamp { Token = token.Token, StartTime = token.StartTime, EndTime = token.EndTime };
        }

        private static NemoForcedAligner.WordTimestamp CloneWord(NemoForcedAligner.WordTimestamp word, Dictionary<NemoForcedAligner.TokenTimestamp, NemoForcedAligner.TokenTimestamp> tokenMap)
        {
            var clone = new NemoForcedAligner.WordTimestamp { Word = word.Word, StartTime = word.StartTime, EndTime = word.EndTime };
            foreach (var t in word.Tokens)
            {
                if (tokenMap.TryGetValue(t, out var clonedToken))
                {
                    clone.Tokens.Add(clonedToken);
                }
                else
                {
                    clone.Tokens.Add(CloneToken(t));
                }
            }
            return clone;
        }
    }
}
