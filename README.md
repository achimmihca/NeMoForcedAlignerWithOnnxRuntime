# NemoForcedAlignerWithOnnxRuntime

The goal of this project is to run **NeMo Forced Aligner (NFA)** with **C#** using **OnnxRuntime**.

Forced alignment is the process of taking an audio file and its corresponding text transcription and determining the start and end times of each word or phoneme in the audio.

## Project Overview

This project aims to provide a C# implementation of the forced alignment process using NVIDIA's NeMo CTC-based models exported to ONNX format. It leverages several powerful .NET libraries for audio processing and machine learning:

- **NAudio:** For audio loading and resampling.
- **NWaves:** For digital signal processing and Mel-spectrogram extraction.
- **Microsoft.ML.OnnxRuntime:** For running the model inference.
- **System.Numerics.Tensors:** For efficient tensor operations.

## Implementation Details

The implementation follows these high-level steps:
1.  **Model Preparation:** Exporting a NeMo CTC model (e.g., Conformer-CTC) to ONNX and extracting its vocabulary.
2.  **Audio Loading:** Reading and resampling audio to 16kHz Mono.
3.  **Preprocessing:** Extracting 80-bin log-Mel spectrograms.
4.  **Inference:** Running the ONNX model to obtain log-probabilities (logprobs).
5.  **Postprocessing:** Using the Viterbi algorithm to align the transcription tokens with the model outputs to obtain precise timestamps.

For a detailed technical guide, please refer to the [ImplementationPlan.md](ImplementationPlan.md).

## Model Metadata
- **Pretrained Model:** `stt_en_conformer_ctc_small`
- **Input Shape:** `[batch, 80, time]` (80 Mel-spectrogram bins)
  - ['audio_signal_dynamic_axes_1', 80, 'audio_signal_dynamic_axes_2']
- **Output Shape:** `[batch, time, 1025]` (1024 BPE tokens + 1 CTC blank)
  - ['LogSoftmaxlogprobs_dim_0', 'LogSoftmaxlogprobs_dim_1', <TOKEN_COUNT>]
    - where TOKEN_COUNT corresponds to the number of lines in the tokens.txt file
- **Downsampling:** 4x (Model output frames are 4x fewer than input frames)
- **Vocabulary:** 1024 SentencePiece BPE tokens. The CTC blank is at index 1024.
- **Opset Version:** 17

## Project Structure

- `NemoForcedAlignerWithOnnxRuntime/`: Core C# project.
- `TestData/`: Sample audio and text files for testing.
- `ImplementationPlan.md`: Detailed technical plan for the implementation.
