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

## Project Structure

- `NemoForcedAlignerWithOnnxRuntime/`: Core C# project.
- `TestData/`: Sample audio and text files for testing.
- `ImplementationPlan.md`: Detailed technical plan for the implementation.
