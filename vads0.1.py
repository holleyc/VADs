import argparse
import torch
import torchaudio
from silero_vad import VoiceActivityDetector, collect_chunks
import whisper

def vad_whisper_pipeline(waveform: torch.Tensor, sample_rate: int, vad: VoiceActivityDetector, model: whisper.Whisper) -> list:
    """
    Runs VAD on the waveform and transcribes each speech segment using Whisper.
    Returns a list of dicts with start time, end time, and transcribed text.
    """
    # Collect speech chunks
    chunks = collect_chunks(
        vad=vad,
        audio=waveform,
        sample_rate=sample_rate,
        min_speech_duration_ms=250,
        speech_pad_ms=30
    )

    results = []
    for start, end, chunk in chunks:
        # Convert tensor to numpy array for whisper
        audio_np = chunk.squeeze(0).cpu().numpy()
        # Transcribe chunk
        transcription = model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = transcription.get("text", "").strip()
        results.append({
            "start_s": start / sample_rate,
            "end_s": end / sample_rate,
            "text": text
        })
    return results


def load_audio(path: str, target_sr: int = 16000) -> tuple:
    """
    Loads an audio file and resamples to target sample rate if needed.
    Returns waveform tensor and sample rate.
    """
    waveform, sr = torchaudio.load(path)
    # Convert to mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
    return waveform, sr


def main():
    parser = argparse.ArgumentParser(description="VAD + Whisper transcription pipeline.")
    parser.add_argument("input", type=str, help="Path to input audio file (wav, mp3, etc.)")
    parser.add_argument("--model", type=str, default="small", help="Whisper model name (tiny, base, small, medium, large)")
    parser.add_argument("--device", type=str, default=None, help="torch device (e.g., cpu or cuda)")
    args = parser.parse_args()

    # Determine device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VAD
    print("Loading Silero VAD...")
    vad = VoiceActivityDetector(device=device)

    # Load Whisper
    print(f"Loading Whisper model '{args.model}'...")
    model = whisper.load_model(args.model, device=device)

    # Load audio
    print(f"Loading audio from '{args.input}'...")
    waveform, sr = load_audio(args.input)
    print(f"Audio loaded, sample rate = {sr}, waveform shape = {waveform.shape}")

    # Run pipeline
    print("Running VAD + transcription pipeline...")
    segments = vad_whisper_pipeline(waveform, sr, vad, model)

    # Print results
    print("\nTranscription results:")
    for i, seg in enumerate(segments, 1):
        print(f"Segment {i}: {seg['start_s']:.2f}s - {seg['end_s']:.2f}s")
        print(f"    {seg['text']}")

if __name__ == "__main__":
    main()
