import time
import threading
import queue
import logging

import numpy as np
import torch
import whisper
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("RealTimeListener")

class RealTimeListener:
    def __init__(self, model='small', device=None, rate=16000, block=0.2, silence=0.5):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model
        self.sample_rate = rate
        self.block_dur = block
        self.silence_dur = silence
        self.block_size = int(rate * block)
        self.silence_blocks = int(silence / block)

        self.q = queue.Queue()
        self.stop_event = threading.Event()

        self._load_models()
        self._init_stream()

    def _load_models(self):
        start = time.time()
        self.vad_model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        (self.get_speech_timestamps, *_) = utils
        self.vad_model.to(self.device)
        log.info(f"Loaded Silero VAD in {time.time() - start:.2f}s")

        start = time.time()
        self.whisper_model = whisper.load_model(self.model_name, device=self.device)
        log.info(f"Loaded Whisper model '{self.model_name}' in {time.time() - start:.2f}s")

    def _init_stream(self):
        def audio_callback(indata, frames, time_info, status):
            if status:
                log.warning(f"Mic status: {status}")
            self.q.put(indata[:, 0].copy())

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            blocksize=self.block_size,
            callback=audio_callback
        )

    def start(self):
        log.info("Starting audio stream...")
        self.stream.start()

    def stop(self):
        self.stop_event.set()
        self.stream.stop()
        self.stream.close()
        log.info("Audio stream stopped.")

    def listen_once(self, timeout=None) -> str:
        """
        Captures one full spoken utterance and returns the transcribed text.
        Optional timeout (in seconds) to exit if no speech is detected.
        """
        in_utt = False
        utt_buf = []
        silence_count = 0
        start_time = None
        start_wait = time.time()

        log.info("Listening for utterance...")

        while not self.stop_event.is_set():
            try:
                block = self.q.get(timeout=1)
            except queue.Empty:
                if timeout and (time.time() - start_wait > timeout):
                    log.warning("Timeout waiting for audio.")
                    return ""
                continue

            wav_block = torch.from_numpy(block).unsqueeze(0).to(self.device)
            ts = self.get_speech_timestamps(
                wav_block, self.vad_model,
                sampling_rate=self.sample_rate,
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

                if silence_count > self.silence_blocks:
                    audio_np = np.concatenate(utt_buf)
                    duration = len(audio_np) / self.sample_rate
                    timestamp = time.strftime('%H:%M:%S', time.localtime(start_time))
                    log.info(f"<<< Utterance ended at {timestamp}, duration {duration:.2f}s")

                    start_trans = time.time()
                    result = self.whisper_model.transcribe(audio_np, fp16=(self.device == 'cuda'))
                    latency = time.time() - start_trans
                    text = result.get('text', '').strip()
                    log.info(f"[{timestamp}] Transcription: {text} (Latency: {latency:.2f}s)")
                    return text
        return ""
