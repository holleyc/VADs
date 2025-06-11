import argparse
import torch
import torchaudio
import whisper

def vad_whisper_pipeline(wav: torch.Tensor, sample_rate: int, vad_model, get_speech_timestamps) -> list:
    """
    Runs VAD on the waveform and transcribes each speech segment using Whisper.
    Returns a list of dicts with start time, end time, and transcribed text.
    """
    # Detect speech timestamps (in seconds)
    speech_ts = get_speech_timestamps(wav, vad_model, sampling_rate=sample_rate, return_seconds=True)

    results = []
    for ts in speech_ts:
        start, end = ts['start'], ts['end']
        start_sample = int(start * sample_rate)
        end_sample   = int(end   * sample_rate)
        chunk = wav[:, start_sample:end_sample]

        # Whisper wants numpy
        audio_np = chunk.squeeze(0).cpu().numpy()
        transcription = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = transcription.get("text", "").strip()

        results.append({"start_s": start, "end_s": end, "text": text})
    return results

def load_audio(path: str, target_sr: int = 16000) -> tuple:
    wav, sr = torchaudio.load(path)
    # Mono
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    return wav, sr

def main():
    parser = argparse.ArgumentParser(description="VAD + Whisper transcription pipeline.")
    parser.add_argument("input",  help="Path to input audio file")
    parser.add_argument("--model", default="small",
                        help="Whisper model: tiny, base, small, medium, large")
    parser.add_argument("--device", default=None,
                        help="Torch device: cpu or cuda")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load Silero VAD from GitHub ----
    print("Loading Silero VAD model via torch.hub...")
    vad_model, utils = torch.hub.load(
        'snakers4/silero-vad',
        'silero_vad',
        force_reload=False
    )
    get_speech_timestamps, _, read_audio, *_ = utils
    vad_model.to(device)

    # ---- Load Whisper ----
    print(f"Loading Whisper model '{args.model}'...")
    global whisper_model
    whisper_model = whisper.load_model(args.model, device=device)

    # ---- Load Audio ----
    print(f"Loading audio from '{args.input}'...")
    wav, sr = load_audio(args.input)
    wav = wav.to(device)
    print(f"Loaded audio: sr={sr}, shape={wav.shape}")

    # ---- Run Pipeline ----
    print("Running VAD + transcription...")
    segments = vad_whisper_pipeline(wav, sr, vad_model, get_speech_timestamps)

    # ---- Print Results ----
    print("\nTranscription results:")
    for i, seg in enumerate(segments, 1):
        print(f"Segment {i}: {seg['start_s']:.2f}s - {seg['end_s']:.2f}s → {seg['text']}")

if __name__ == "__main__":
    main()
