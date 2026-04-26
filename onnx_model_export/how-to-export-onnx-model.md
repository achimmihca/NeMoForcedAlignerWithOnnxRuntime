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
# Run the export script for the default English model
python export_nfa_onnx.py

# Run for other languages
python export_nfa_onnx.py --model stt_de_conformer_ctc_large
python export_nfa_onnx.py --model stt_es_conformer_ctc_large
python export_nfa_onnx.py --model stt_fr_conformer_ctc_large

# Run for a multilingual model (Canary)
python export_nfa_onnx.py --model canary-1b
```

### 5. Multilingual Alternatives
The following models are recommended for forced alignment. 

| Language | Model Name (Large) | Model Name (Small/Medium) |
| :--- | :--- | :--- |
| **English** | `stt_en_conformer_ctc_large` | `stt_en_conformer_ctc_small` (Default) |
| **German** | `stt_de_conformer_ctc_large` | `stt_de_citrinet_1024` (Medium) |
| **Spanish** | `stt_es_conformer_ctc_large` | - |
| **French** | `stt_fr_conformer_ctc_large` | `stt_fr_quartznet15x5` (Small) |
| **Multilingual** | `canary-1b` | `canary-1b-flash` (Faster/Smaller) |

*Note: If a specific `_small` Conformer model is not available for a language, you can try Citrinet or QuartzNet models which are generally smaller and faster, though potentially less accurate than Conformer Large.*

### 6. Summary of Inputs/Outputs
- **Model Source:** Automatically downloaded by the script from NVIDIA NGC or Hugging Face.
- **Inputs for the script:** Optional `--model` and `--output` arguments.
- **Outputs generated:**
    - `{model_name}.onnx`: The exported model for use in C#.
    - `tokens_{model_name}.txt`: The vocabulary file used for decoding.
- **Audio Requirements (for C#):** The model expects 16kHz, Mono, 16-bit PCM WAV files.

### Note for Windows Users
If you encounter errors regarding `pynini` or `nemo_text_processing`, these are often caused by native library requirements. For simple ONNX export of a CTC model, these can usually be ignored or bypassed by ensuring you have the latest `pip` and `setuptools` installed. If issues persist, consider running the export in **WSL2 (Ubuntu)**.