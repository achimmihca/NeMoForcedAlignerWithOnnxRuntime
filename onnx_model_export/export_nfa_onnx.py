import nemo.collections.asr as nemo_asr
import os

def export_model():
    model_name = "stt_en_conformer_ctc_small"
    onnx_file = "nfa_model.onnx"
    tokens_file = "tokens.txt"

    print(f"Loading pretrained model: {model_name}...")
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=model_name)

    print(f"Exporting model to {onnx_file}...")
    model.export(onnx_file)

    print(f"Extracting vocabulary to {tokens_file}...")
    tokens = model.decoder.vocabulary
    with open(tokens_file, "w", encoding="utf-8") as f:
        for token in tokens:
            f.write(token + "\n")

    print("Done!")

if __name__ == "__main__":
    export_model()
