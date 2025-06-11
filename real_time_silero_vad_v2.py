"""Real-time Silero VAD with buffering and debug prints"""
import torch
import sounddevice as sd
import numpy as np
import argparse

# Load Silero VAD
device = "cuda" if torch.cuda.is_available() else "cpu"
vad_model, utils = torch.hub.load(
    'snakers4/silero-vad',
    'silero_vad',
    force_reload=False
)
get_speech_timestamps, _, read_audio, *_ = utils
vad_model.to(device)

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.2  # seconds per block
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

def main():
    parser = argparse.ArgumentParser(description="Real-time Silero VAD with buffer")
    parser.add_argument('--buffer_sec', type=float, default=1.0,
                        help="Size of rolling buffer in seconds")
    args = parser.parse_args()

    buffer_size = int(SAMPLE_RATE * args.buffer_sec)
    buffer = np.zeros(buffer_size, dtype=np.float32)
    write_ptr = 0

    print(f"Using device: {device}")
    print(f"Listening for voice... block {BLOCK_DURATION}s, buffer {args.buffer_sec}s")
    
    def callback(indata, frames, time, status):
        nonlocal buffer, write_ptr
        if status:
            print(f"Status: {status}", flush=True)
        # mono float32
        audio = indata[:,0].astype(np.float32)
        # debug amplitude
        print(f"Block amplitude min {audio.min():.3f}, max {audio.max():.3f}", flush=True)
        # write into rolling buffer
        n = len(audio)
        if n >= buffer_size:
            buffer = audio[-buffer_size:]
            write_ptr = 0
        else:
            end = write_ptr + n
            if end <= buffer_size:
                buffer[write_ptr:end] = audio
            else:
                part1 = buffer_size - write_ptr
                buffer[write_ptr:] = audio[:part1]
                buffer[:n-part1] = audio[part1:]
            write_ptr = end % buffer_size
        
        # run VAD on full buffer
        wav = torch.from_numpy(buffer).unsqueeze(0).to(device)
        ts = get_speech_timestamps(wav, vad_model, sampling_rate=SAMPLE_RATE, return_seconds=True)
        if ts:
            print("🗣️ Voice detected in buffer", flush=True)
        else:
            print("🤐 Silence", flush=True)

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE, callback=callback):
        try:
            sd.sleep(int(1e9))
        except KeyboardInterrupt:
            print("\nStopped listening.")

if __name__ == "__main__":
    main()
