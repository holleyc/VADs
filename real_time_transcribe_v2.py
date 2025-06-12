import argparse
import time
import queue
import threading
import logging
import datetime

import torch
import whisper
import numpy as np
import sounddevice as sd

# ────────────────────── Logging Setup ────────────────────── #
# Console logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger("VAD-Whisper")

# File handler for daily logs
def setup_daily_file_handler():
    """
    Creates or appends to a daily log file named transcripts-YYYY-MM-DD.log
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    filename = f"transcripts-{today}.log"
    fh = logging.FileHandler(filename, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S'))
    log.addHandler(fh)

# Initialize file handler
setup_daily_file_handler()

# ────────────────────── Config & Args ────────────────────── #
def parse_args():
    p = argparse.ArgumentParser("Real-time VAD + Whisper")
    p.add_argument('--model',    default='small', help='Whisper model (locked to English)')
    p.add_argument('--device',   default=None,    help='torch device: cpu or cuda')
    p.add_argument('--rate',     type=int, default=16000, help='Sampling rate')
    p.add_argument('--block',    type=float, default=0.2, help='Block duration (s)')
    p.add_argument('--silence',  type=float, default=0.5, help='Min silence (s) to end utterance')
    return p.parse_args()

# ────────────────────── Transcription Callback ────────────────────── #
def on_transcription(text, timestamp, meta=None):
    log.info(f"[{timestamp}] {text}")
    if meta:
        log.debug(f"Meta: {meta}")

# ────────────────────── Main Transcription Engine ────────────────────── #
def run(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_RATE = args.rate
    BLOCK_DUR = args.block
    SILENCE_DUR = args.silence
    BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DUR)
    SILENCE_BLOCKS = int(SILENCE_DUR / BLOCK_DUR)

    log.info(f"Device={device}, rate={SAMPLE_RATE}, block={BLOCK_DUR}s, silence={SILENCE_DUR}s")

    load_start = time.time()
    vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
    get_speech_timestamps, *_ = utils
    vad_model.to(device)
    vad_time = time.time() - load_start
    log.info(f"Loaded Silero VAD in {vad_time:.2f}s")

    load_start = time.time()
    whisper_model = whisper.load_model(args.model, device=device)
    whisper_time = time.time() - load_start
    log.info(f"Loaded Whisper model '{args.model}' in {whisper_time:.2f}s")

    q = queue.Queue()
    stop_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        if status:
            log.warning(f"Mic status: {status}")
        if indata is not None:
            q.put(indata[:, 0].copy())
        else:
            log.warning("No input data in callback!")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1,
        blocksize=BLOCK_SIZE, callback=audio_callback
    )
    stream.start()
    log.info("Listening... Press Ctrl+C to stop.")

    in_utt = False
    utt_buf = []
    silence_count = 0
    utt_count = 0
    start_time = None
    total_audio_sec = 0
    total_transcribe_time = 0
    total_utterances = 0

    try:
        while not stop_event.is_set():
            q_start = time.time()
            block = q.get()
            q_latency = time.time() - q_start

            wav_block = torch.from_numpy(block).unsqueeze(0).to(device)
            vad_start = time.time()
            ts = get_speech_timestamps(
                wav_block, vad_model,
                sampling_rate=SAMPLE_RATE,
                threshold=0.7,
                min_speech_duration_ms=200,
                min_silence_duration_ms=200,
                speech_pad_ms=50,
                return_seconds=False
            )
            vad_latency = time.time() - vad_start

            is_speech = len(ts) > 0
            log.debug(f"Block latency: queue={q_latency:.3f}s vad={vad_latency:.3f}s speech={is_speech}")

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

                    transcribe_start = time.time()
                    # Force English-only transcription
                    result = whisper_model.transcribe(
                        audio_np,
                        language='en',
                        task='transcribe',
                        fp16=(device == 'cuda')
                    )
                    transcribe_latency = time.time() - transcribe_start

                    total_audio_sec += duration
                    total_transcribe_time += transcribe_latency
                    total_utterances += 1

                    text = result.get('text', '').strip()
                    on_transcription(text, timestamp, meta={
                        "duration_sec": duration,
                        "transcribe_time_sec": transcribe_latency,
                        "throughput_x_real_time": duration / transcribe_latency if transcribe_latency > 0 else 0
                    })

    except KeyboardInterrupt:
        stop_event.set()
        log.info("Stopping by user interrupt.")
    finally:
        stream.stop()
        stream.close()
        log.info("Stream closed.")

        if total_utterances > 0:
            avg_time = total_transcribe_time / total_utterances
            avg_rt_factor = total_audio_sec / total_transcribe_time if total_transcribe_time > 0 else 0
            log.info(f"Summary: {total_utterances} utterances | "
                     f"{total_audio_sec:.1f}s audio | "
                     f"{total_transcribe_time:.1f}s transcription time | "
                     f"Avg latency: {avg_time:.2f}s | Real-time factor: {avg_rt_factor:.2f}x")

# ────────────────────── Entrypoint ────────────────────── #
if __name__ == '__main__':
    run(parse_args())
