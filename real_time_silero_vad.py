"""Real-time Silero VAD voice activity detection demo"""
import torch
import sounddevice as sd
import numpy as np

# Load Silero VAD model from GitHub
device = "cuda" if torch.cuda.is_available() else "cpu"
vad_model, utils = torch.hub.load(
    'snakers4/silero-vad',
    'silero_vad',
    force_reload=False
)
get_speech_timestamps, _, read_audio, *_ = utils
vad_model.to(device)

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.03  # 30 ms
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

print(f"Using device: {device}")
print("Listening for voice activity (Press Ctrl+C to stop)...")

def callback(indata, frames, time, status):
    if status:
        print(f"Status: {status}", flush=True)
    # Convert to torch tensor
    audio = torch.from_numpy(indata[:, 0].astype(np.float32)).unsqueeze(0).to(device)
    # Detect speech timestamps in this block
    timestamps = get_speech_timestamps(audio, vad_model, sampling_rate=SAMPLE_RATE, return_seconds=True)
    if timestamps:
        print("🗣️  Voice detected", flush=True)
    else:
        print("🤐  Silence", flush=True)

try:
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=callback):
        sd.sleep(int(1e9))  # Run until interrupted
except KeyboardInterrupt:
    print("\nStopped listening.")
