To use the `export_nfa_onnx.py` script, you need to set up a Python environment with the NVIDIA NeMo toolkit. The script itself handles the downloading of the model automatically.

### 1. Prerequisites
You need **Python 3.8 to 3.10** installed on your system.

### 2. Set up a Virtual Environment
It is recommended to use a virtual environment to avoid dependency conflicts.

```powershell
# Create a virtual environment
python -m venv nfa_env

# Activate the environment
.\nfa_env\Scripts\activate
```

### 3. Install Dependencies
Install the required packages. Note that `nemo_toolkit[asr]` includes the necessary logic for Conformer models.

```powershell
# Install PyTorch (CPU version is sufficient for export)
pip install torch torchvision torchaudio

# Install NeMo ASR collection and ONNX export tools
pip install nemo_toolkit[asr]
pip install onnx onnxruntime
```

### 4. Download and Export
You do **not** need to download the model file manually. When you run the script, NeMo's `from_pretrained` method will automatically download the original model from the NVIDIA NGC registry to your local cache.

```powershell
# Run the export script
python export_nfa_onnx.py
```

### 5. Summary of Inputs/Outputs
- **Model Source:** Automatically downloaded by the script from NVIDIA NGC (name: `stt_en_conformer_ctc_small`).
- **Inputs for the script:** None (it pulls from the cloud).
- **Outputs generated:**
    - `nfa_model.onnx`: The exported model for use in C#.
    - `tokens.txt`: The vocabulary file used for decoding.
- **Audio Requirements (for C#):** The model expects 16kHz, Mono, 16-bit PCM WAV files.

### Note for Windows Users
If you encounter errors regarding `pynini` or `nemo_text_processing`, these are often caused by native library requirements. For simple ONNX export of a CTC model, these can usually be ignored or bypassed by ensuring you have the latest `pip` and `setuptools` installed. If issues persist, consider running the export in **WSL2 (Ubuntu)**.