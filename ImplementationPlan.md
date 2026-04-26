# Implementation Plan: NeMo Forced Aligner (NFA) with C# and OnnxRuntime

This document outlines the steps to implement forced alignment using NeMo's CTC-based models in C# with OnnxRuntime.

## 1. Prepare NFA ONNX Model

NeMo models (typically `.nemo` files) need to be exported to ONNX format.

### Steps:
- [x] **Export from Python:**
  Use NeMo's export functionality. A Conformer-CTC model (e.g., `stt_en_conformer_ctc_small`) is recommended.
  ```python
  import nemo.collections.asr as nemo_asr
  model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name="stt_en_conformer_ctc_small")
  model.export("nfa_model.onnx")
  ```
- [x] **Extract Vocabulary:**
  Extract the vocabulary/tokens from the model to map model output indices back to characters/subwords.
  ```python
  tokens = model.decoder.vocabulary
  with open("tokens.txt", "w") as f:
      for token in tokens:
          f.write(token + "\n")
  ```
- [x] **Identify Input/Output Nodes:**
  Use a tool like Netron to verify input names (usually `audio_signal` and `length`) and output names (usually `logprobs`).
  - **Input:** `audio_signal` shape `[batch, 80, time]` (80 Mel bins).
  - **Output:** `logprobs` shape `[batch, time, 1025]` (1024 BPE tokens + 1 CTC blank).
  - **Opset Version:** 17.
  - **Downsampling:** The output time dimension is 4x smaller than the input (Conformer-CTC stride).

## 2. Load Audio with NAudio

Use `NAudio` to read the `.wav` file and ensure it is in the format expected by the model (typically 16kHz, Mono, 32-bit float).

### Implementation:
- Open `AudioFileReader`.
- If the sample rate is not 16000Hz, use `MediaFoundationResampler` or `WavChannel32` to resample.
- Read samples into a `float[]` array.

## 3. Preprocess Audio with NWaves

NeMo Conformer models expect Mel-spectrogram features as input.

### Configuration:
- **Sample Rate:** 16000 Hz
- **Window Size:** 25ms (400 samples)
- **Hop Size:** 10ms (160 samples)
- **Window Function:** Hann
- **Features:** 80 Mel bins
- **Frequency Range:** 0 to 8000 Hz

### Steps:
- Use `NWaves.FeatureExtractors.MelSpectrogramExtractor`.
- Extract features from the audio samples.
- Apply Log-scaling: `log(mel_spectrogram + 1e-5)`.
- (Optional) Normalize features (Mean and Variance normalization) if required by the specific model.

## 4. Prepare Tensor Inputs

Convert the preprocessed features into OnnxRuntime `OrtValue` or `DenseTensor`.

### Inputs:
- **audio_signal:** A 3D tensor of shape `[batch_size, feature_dim, sequence_length]`. 
  - Note: NeMo usually expects features as `[1, 80, T]`.
- **length:** A 1D tensor `[batch_size]` containing the sequence length `T`.

## 5. Run Inference with OnnxRuntime

Execute the model using `InferenceSession`.

### Implementation:
- Initialize `InferenceSession` with the `nfa_model.onnx`.
- Create `NamedOnnxValue` for `audio_signal` and `length`.
- Run `session.Run(inputs)` to get the `logprobs`.
- Output `logprobs` shape will be `[batch_size, sequence_length, vocabulary_size]`.

## 6. Postprocess Tensor Outputs (Forced Alignment)

Align the transcription tokens with the `logprobs` using the Viterbi algorithm.

### Steps:
- **Tokenization:** Convert the reference text into token IDs using the extracted `tokens.txt`.
  - Indices 0-1023: BPE tokens from `tokens.txt`.
  - Index 1024: CTC Blank token.
  - Note: Tokens starting with ` ` (Unicode `U+2581`) indicate a space/new word.
- **Viterbi Decoding:** 
  - Implement the CTC-style Viterbi alignment.
  - Constrain the path to match the sequence of tokens in the reference text.
  - Account for the "blank" token (usually index 0 or last index, check vocabulary).
- **Timestamp Calculation:**
  - Map the frame indices from the Viterbi path back to time.
  - `time = frame_index * hop_length / sample_rate`.
  - Account for the model's downsampling factor (Conformer-CTC usually has a 4x or 8x downsampling).

## Summary of Libraries
- **NAudio:** Audio IO and resampling.
- **NWaves:** DSP and Mel-spectrogram extraction.
- **Microsoft.ML.OnnxRuntime:** Model inference.
- **System.Numerics.Tensors:** Tensor data structures.
