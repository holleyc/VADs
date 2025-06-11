"""
Real-time Silero VAD → Whisper transcription:

 1. Listens to the microphone in blocks.
 2. Uses Silero VAD to detect speech blocks.
 3. Buffers an utterance while speech continues.
 4. On end-of-speech, sends the buffered audio to Whisper for transcription.
 5. Prints timestamped transcriptions.

Requirements:
  pip install torch whisper sounddevice numpy

Usage:
  python3 real_time_transcribe.py \
    --model small \
    --device cuda \
    --rate 16000 \
    --block 0.2 \
    --silence 0.5
"""
import argparse
import time
import queue
import threading

import torch
import whisper
import numpy as np
import sounddevice as sd

def int_or_default(x, default):
    try:
        return int(x)
    except:
        return default

def main():
    p = argparse.ArgumentParser("Real-time VAD + Whisper")
    p.add_argument('--model',    default='small', help='Whisper model')
    p.add_argument('--device',   default=None,    help='torch device: cpu or cuda')
    p.add_argument('--rate',     type=int, default=16000,  help='Sampling rate')
    p.add_argument('--block',    type=float, default=0.2,   help='Block duration in seconds')
    p.add_argument('--silence',  type=float, default=0.5,   help='Min silence (s) to end utterance')
    args = p.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_RATE = args.rate
    BLOCK_DUR   = args.block
    SILENCE_DUR = args.silence
    BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_DUR)
    SILENCE_BLOCKS = int(SILENCE_DUR / BLOCK_DUR)

    print(f"Device={device}, rate={SAMPLE_RATE}, block={BLOCK_DUR}s, silence threshold={SILENCE_DUR}s")

    # Load VAD
    vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    get_speech_timestamps, _, _, *_ = utils
    vad_model.to(device)

    # Load Whisper
    model = whisper.load_model(args.model, device=device)

    # Thread-safe queue for audio blocks
    q = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Mic status:", status)
        # flatten to 1D float32 numpy
        q.put(indata[:,0].copy())

    # Start audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        blocksize=BLOCK_SIZE, callback=audio_callback
    )
    stream.start()
    print("Listening... Press Ctrl+C to stop.")

    # State for utterance detection
    in_utt = False
    utt_buf = []
    silence_count = 0
    utt_count = 0
    start_time = None

    try:
        while not stop_event.is_set():
            block = q.get()  # numpy float32 array of length BLOCK_SIZE
            # convert to torch tensor [1, N]
            wav_block = torch.from_numpy(block).unsqueeze(0).to(device)
            ts = get_speech_timestamps(
                wav_block, vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=0.7,
                min_speech_duration_ms=200,
                min_silence_duration_ms=200,
                speech_pad_ms=50,
                return_seconds=False
            )

            is_speech = len(ts) > 0

            if is_speech and not in_utt:
                in_utt = True
                utt_buf = [block]
                silence_count = 0
                start_time = time.time()
                print(">>> Utterance started")

            elif in_utt:
                utt_buf.append(block)
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1

                if silence_count > SILENCE_BLOCKS:
                    # end utterance
                    in_utt = False
                    utt_count += 1
                    audio_np = np.concatenate(utt_buf)
                    duration = len(audio_np) / SAMPLE_RATE
                    timestamp = time.strftime('%H:%M:%S', time.localtime(start_time))
                    print(f"<<< Utterance {utt_count} ended at {timestamp}, duration {duration:.2f}s")

                    # Transcribe
                    result = model.transcribe(audio_np, fp16=(device=='cuda'))
                    text = result.get('text','').strip()
                    print(f"[{timestamp}] {text}\n")

            # else: not in utterance and no speech => continue
    except KeyboardInterrupt:
        stop_event.set()
        print("Stopping...")
    finally:
        stream.stop()
        stream.close()

if __name__ == '__main__':
    main()