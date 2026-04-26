import os
import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.utils.aligner_utils import viterbi_decoding, get_utt_obj, add_t_start_end_to_utt_obj, Word, Token, Segment
import librosa
import numpy as np
import json

def run_forced_alignment(audio_path, transcript_path, model_name, output_json_path):
    print(f"Audio: {audio_path}")
    
    # 1. Load Model
    print(f"Loading model: {model_name}...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 2. Load Audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(device)
    audio_len = torch.tensor([len(audio)]).to(device)

    # 3. Get Log-Probs
    with torch.no_grad():
        log_probs, encoded_len, _ = model(input_signal=audio_tensor, input_signal_length=audio_len)
    
    # 4. Get Transcript
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read().replace("\n", " ").strip()
    print(f"Transcript: {transcript}")

    # 5. Prepare Utterance Object
    num_frames = log_probs.shape[1]
    utt_obj = get_utt_obj(
        text=transcript,
        T=num_frames,
        model=model,
        utt_id="test_utt"
    )
    
    y_batch = torch.tensor([utt_obj.token_ids_with_blanks]).to(device)
    u_batch = torch.tensor([len(utt_obj.token_ids_with_blanks)]).to(device)
    
    # 6. Viterbi Decoding
    alignments = viterbi_decoding(log_probs, y_batch, encoded_len, u_batch)
    token_frames = alignments[0]
    
    # 7. Calculate Timestamps
    # alignments[0] contains the index of the token aligned to each frame
    audio_duration = len(audio) / sr
    frame_duration = audio_duration / encoded_len[0].item()
    
    # Use the provided NeMo function to add start/end times to the utt_obj
    add_t_start_end_to_utt_obj(utt_obj, token_frames, frame_duration)
    
    # 8. Print and Collect Results
    print(f"\nWord Timestamps:")
    results = []
    for item in utt_obj.segments_and_tokens:
        if isinstance(item, Segment):
            for sub_item in item.words_and_tokens:
                if isinstance(sub_item, Word):
                    print(f"Word: {sub_item.text.ljust(15)} | Start: {sub_item.t_start:.3f}s | End: {sub_item.t_end:.3f}s")
                    results.append({
                        "word": sub_item.text,
                        "start": round(float(sub_item.t_start), 3),
                        "end": round(float(sub_item.t_end), 3)
                    })

    # 9. Export to JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults exported to {output_json_path}")

    print(f"Forced alignment successful!")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_data_dir = os.path.join(project_root, "NemoForcedAlignerWithOnnxRuntime", "TestData")
    test_result_dir = os.path.join(project_root, "onnx_model_export", "TestResult")
    
    audio_file = "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
    transcript_file = "Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.txt"
    model_name = "stt_en_conformer_ctc_small"
    
    audio_path = os.path.join(test_data_dir, audio_file)
    transcript_path = os.path.join(test_data_dir, transcript_file)
    output_json_path = os.path.join(test_result_dir, "alignment_result.json")
    
    if os.path.exists(audio_path) and os.path.exists(transcript_path):
        run_forced_alignment(audio_path, transcript_path, model_name, output_json_path)
    else:
        print("Test data files not found.")

if __name__ == "__main__":
    main()
