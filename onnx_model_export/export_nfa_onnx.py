import nemo.collections.asr as nemo_asr
import os
import argparse

def export_model():
    parser = argparse.ArgumentParser(description="Export NeMo ASR model to ONNX")
    parser.add_argument("--model", type=str, default="stt_en_conformer_ctc_small", help="Pretrained model name (e.g., stt_de_conformer_ctc_large)")
    parser.add_argument("--output", type=str, default=None, help="Output ONNX file name")
    args = parser.parse_args()

    model_name = args.model
    onnx_file = args.output if args.output else f"{model_name.replace('/', '_')}.onnx"
    tokens_file = f"tokens_{model_name.replace('/', '_')}.txt"

    print(f"Loading pretrained model: {model_name}...")
    # Using ASRModel.from_pretrained to support various model types (CTC, Hybrid)
    try:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        print("Falling back to EncDecCTCModelBPE...")
        model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)

    # For Hybrid models (like Canary), we want to export the CTC decoder
    if hasattr(model, "set_export_config"):
        print("Setting export config to ctc...")
        model.set_export_config({"decoder_type": "ctc"})

    print(f"Exporting model to {onnx_file}...")
    model.export(onnx_file)

    print(f"Extracting vocabulary to {tokens_file}...")
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
        tokens = model.decoder.vocabulary
    elif hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'vocab'):
        # Some models might have tokenizer instead of decoder.vocabulary
        tokens = model.tokenizer.vocab
    else:
        # Generic way to get labels
        tokens = model.cfg.decoder.vocabulary if hasattr(model.cfg, 'decoder') else model.cfg.labels

    with open(tokens_file, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")

    print("Done!")

if __name__ == "__main__":
    export_model()
