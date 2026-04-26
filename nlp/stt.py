"""
stt.py - Speech-to-Text sử dụng Vosk offline cho tiếng Việt
"""

import json
import queue
import sounddevice as sd
import vosk
import sys
import os
from typing import Optional, Callable

from .config import NLPConfig


class VoskSTT:
    """Nhận dạng giọng nói tiếng Việt offline sử dụng Vosk"""
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.model = None
        self.recognizer = None
        self.audio_queue = queue.Queue()
        self.is_listening = False
        
        self._load_model()
    
    def _load_model(self):
        """Load model Vosk"""
        model_path = self.config.VOSK_MODEL_PATH
        
        if not os.path.exists(model_path):
            print(f"[Vosk] Model not found at {model_path}")
            print("[Vosk] Please download from: https://alphacephei.com/vosk/models")
            print("[Vosk] Download: vosk-model-small-vi-0.22.zip and extract to models/")
            raise FileNotFoundError(f"Vosk model not found: {model_path}")
        
        print(f"[Vosk] Loading model from {model_path}...")
        self.model = vosk.Model(model_path)
        self.recognizer = vosk.KaldiRecognizer(self.model, self.config.SAMPLE_RATE)
        print("[Vosk] Model loaded successfully")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback xử lý audio từ microphone"""
        if status:
            print(f"[Vosk] Audio status: {status}")
        self.audio_queue.put(bytes(indata))
    
    def listen_once(self, timeout: float = 5.0) -> Optional[str]:
        """
        Lắng nghe một câu lệnh từ microphone
        
        Args:
            timeout: Thời gian chờ tối đa (giây)
        
        Returns:
            text: Câu lệnh dạng text, hoặc None nếu timeout
        """
        import numpy as np
        
        self.is_listening = True
        text_result = None
        
        try:
            # Mở microphone stream
            stream = sd.RawInputStream(
                samplerate=self.config.SAMPLE_RATE,
                blocksize=8000,
                device=None,  # device mặc định
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            )
            
            print("[Vosk] Listening... (nói câu lệnh)")
            stream.start()
            
            # Thu thập audio cho đến khi có kết quả hoặc timeout
            import time
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                try:
                    data = self.audio_queue.get(timeout=0.5)
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '')
                        if text:
                            text_result = text
                            print(f"[Vosk] Recognized: {text}")
                            break
                except queue.Empty:
                    continue
            
            stream.stop()
            stream.close()
            
        except Exception as e:
            print(f"[Vosk] Error: {e}")
            # Fallback: dùng input() cho testing
            text_result = input("[TEST MODE] Nhập câu lệnh: ")
        
        finally:
            self.is_listening = False
        
        return text_result
    
    def listen_continuous(self, callback: Callable[[str], None]):
        """Lắng nghe liên tục, gọi callback khi có lệnh"""
        import time
        
        self.is_listening = True
        
        try:
            stream = sd.RawInputStream(
                samplerate=self.config.SAMPLE_RATE,
                blocksize=8000,
                device=None,
                dtype='int16',
                channels=1,
                callback=self.audio_callback
            )
            
            print("[Vosk] Continuous listening mode... (press Ctrl+C to stop)")
            stream.start()
            
            while self.is_listening:
                try:
                    data = self.audio_queue.get(timeout=0.5)
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        text = result.get('text', '')
                        if text:
                            callback(text)
                except queue.Empty:
                    continue
                    
        except KeyboardInterrupt:
            print("\n[Vosk] Stopped listening")
        finally:
            self.is_listening = False
            if 'stream' in locals():
                stream.stop()
                stream.close()


# Mock STT cho testing (không cần microphone)
class MockSTT:
    """Mock STT dùng input() để test không cần microphone"""
    
    def listen_once(self, timeout: float = 5.0) -> str:
        text = input("[Mock] Nhập câu lệnh: ")
        return text if text else None
    
    def listen_continuous(self, callback):
        while True:
            text = input("[Mock] Nhập lệnh (q để thoát): ")
            if text == 'q':
                break
            if text:
                callback(text)
