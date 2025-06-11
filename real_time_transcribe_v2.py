import argparse
import time
import queue
import threading
import logging

import torch
import whisper
import numpy as np
import sounddevice as sd

# ────────────────────── Logging Setup ────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("VAD-Whisper")

# ────────────────────── Config & Args ────────────────────── #
def parse_args():
    p = argparse.ArgumentParser("Real-time VAD + Whisper")
    p.add_argument('--model',    default='small', help='Whisper model')
    p.add_argument('--device',   default=None,    help='torch device: cpu or cuda')
    p.add_argument('--rate',     type=int, default=16000, help='Sampling rate')
    p.add_argument('--block',    type=float, default=0.2, help='Block duration (s)')
    p.add_argument('--silence',  type=float, default=0.5, help='Min silence (s) to end utterance')
    return p.parse_args()

# ────────────────────── Transcription Callback ────────────────────── #
def on_transcription(text, timestamp):
    log.info(f"[{timestamp}] {text}")
    # Future hook: push to LLM, WebSocket, REST API, etc.

# ────────────────────── Main Transcription Engine ────────────────────── #
def run(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_RATE = args.rate
    BLOCK_DUR = args.block
    SILENCE_DUR = args.silence
    BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DUR)
    SILENCE_BLOCKS = int(SILENCE_DUR / BLOCK_DUR)

    log.info(f"Device={device}, rate={SAMPLE_RATE}, block={BLOCK_DUR}s, silence={SILENCE_DUR}s")

    # Load models
    vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    get_speech_timestamps, *_ = utils
    vad_model.to(device)

    whisper_model = whisper.load_model(args.model, device=device)

    q = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            log.warning(f"Mic status: {status}")
        q.put(indata[:, 0].copy())

    # Audio stream
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        blocksize=BLOCK_SIZE, callback=audio_callback
    )
    stream.start()
    log.info("Listening... Press Ctrl+C to stop.")

    # Utterance state
    in_utt = False
    utt_buf = []
    silence_count = 0
    utt_count = 0
    start_time = None

    try:
        while not stop_event.is_set():
            block = q.get()
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
                log.info(">>> Utterance started")

            elif in_utt:
                utt_buf.append(block)
                silence_count = 0 if is_speech else silence_count + 1

                if silence_count > SILENCE_BLOCKS:
                    in_utt = False
                    utt_count += 1
                    audio_np = np.concatenate(utt_buf)
                    duration = len(audio_np) / SAMPLE_RATE
                    timestamp = time.strftime('%H:%M:%S', time.localtime(start_time))
                    log.info(f"<<< Utterance {utt_count} ended at {timestamp}, duration {duration:.2f}s")

                    result = whisper_model.transcribe(audio_np, fp16=(device == 'cuda'))
                    text = result.get('text', '').strip()
                    on_transcription(text, timestamp)

    except KeyboardInterrupt:
        stop_event.set()
        log.info("Stopping...")
    finally:
        stream.stop()
        stream.close()

# ────────────────────── Entrypoint ────────────────────── #
if __name__ == '__main__':
    run(parse_args())
