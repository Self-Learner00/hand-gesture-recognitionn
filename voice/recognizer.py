import threading
import queue
import time
from typing import Optional, Callable
from commands.command_types import CommandEvent, CommandType
from commands.command_mapping import CommandMapping
from voice.command_parser import VoiceCommandParser
from utils.config import ConfigManager, config_manager
from utils.logger import system_logger

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except Exception:
    sr = None
    _SR_AVAILABLE = False

class VoiceRecognizer:
    """
    Background worker thread for continuous, non-blocking speech recognition.
    Resilient to microphone disconnections, permission errors, and audio driver failures.
    Pushes detected CommandEvents to a thread-safe queue.
    """
    def __init__(self, command_mapping: CommandMapping, cfg_mgr: Optional[ConfigManager] = None):
        self.command_mapping = command_mapping
        self.cfg_mgr = cfg_mgr or config_manager
        self.parser = VoiceCommandParser(command_mapping, self.cfg_mgr)

        self.command_queue: queue.Queue[CommandEvent] = queue.Queue()
        self.running: bool = False
        self.is_listening: bool = False
        self.is_mic_ready: bool = False
        self.mic_status: str = "INITIALIZING"
        self.last_transcript: str = ""
        self.last_parsed_command: CommandType = CommandType.NONE
        self.last_status_message: str = "Ready"
        self.error_count: int = 0

        self.thread: Optional[threading.Thread] = None
        self.recognizer = None
        self.microphone = None

        if _SR_AVAILABLE:
            self.recognizer = sr.Recognizer()
            self.recognizer.energy_threshold = self.cfg_mgr.get("voice", "energy_threshold", default=300)
            self.recognizer.dynamic_energy_threshold = self.cfg_mgr.get("voice", "dynamic_energy_threshold", default=True)
            self.recognizer.pause_threshold = self.cfg_mgr.get("voice", "pause_threshold", default=0.8)

    def _setup_microphone(self) -> bool:
        """Attempts to initialize the microphone."""
        if not _SR_AVAILABLE:
            self.mic_status = "SR_UNAVAILABLE"
            return False

        try:
            # Check available input devices
            mics = sr.Microphone.list_microphone_names()
            if not mics:
                self.mic_status = "NO_MIC_HARDWARE"
                system_logger.warning("No microphone devices detected by SpeechRecognition.")
                return False

            # Check if specific mic index or earbuds preference is requested
            device_index = self.cfg_mgr.get("voice", "device_index", default=None)
            prefer_earbuds = self.cfg_mgr.get("voice", "prefer_earbuds", default=True)

            target_idx = None
            is_earbuds = False

            if device_index is not None and isinstance(device_index, int) and 0 <= device_index < len(mics):
                target_idx = device_index
            elif prefer_earbuds:
                # Find Bluetooth / Earbud device automatically
                keywords = ["bluez_input", "bluetooth", "earbud", "buds", "airpod", "headset"]
                for i, name in enumerate(mics):
                    name_lower = name.lower()
                    if any(kw in name_lower for kw in keywords):
                        target_idx = i
                        is_earbuds = True
                        break

            if target_idx is not None:
                self.microphone = sr.Microphone(device_index=target_idx)
                dev_name = mics[target_idx]
                if is_earbuds or "bluez" in dev_name.lower():
                    self.mic_status = "EARBUDS (BT)"
                else:
                    self.mic_status = f"MIC [{target_idx}]"
                system_logger.info(f"Using designated microphone device [{target_idx}]: {dev_name}")
            else:
                self.microphone = sr.Microphone()
                self.mic_status = "CONNECTED"

            with self.microphone as source:
                # Quick calibration with short duration so startup is fast
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

            self.is_mic_ready = True
            system_logger.info(f"Microphone ({self.mic_status}) successfully initialized and calibrated.")
            return True

        except Exception as e:
            self.is_mic_ready = False
            self.mic_status = f"MIC_ERROR: {type(e).__name__}"
            system_logger.warning(f"Microphone initialization failed ({e}). Running in voice simulation/injection mode.")
            self.microphone = None
            return False

    def start(self):
        """Starts the background listening thread."""
        if self.running:
            return

        enabled = self.cfg_mgr.get("voice", "enabled", default=True)
        if not enabled:
            self.mic_status = "DISABLED_IN_CONFIG"
            system_logger.info("Voice recognition disabled in configuration.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="VoiceRecognizerThread")
        self.thread.start()
        system_logger.info("Voice recognizer background thread started.")

    def stop(self):
        """Stops the listening loop and releases resources."""
        self.running = False
        self.is_listening = False
        self.is_mic_ready = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.mic_status = "STOPPED"
        system_logger.info("Voice recognizer stopped.")

    def inject_phrase(self, phrase: str) -> Optional[CommandEvent]:
        """
        Manually injects a spoken phrase for testing, fallback UI interaction, or automated tests.
        """
        self.last_transcript = phrase
        cmd_event = self.parser.parse(phrase)
        if cmd_event:
            self.last_parsed_command = cmd_event.command
            self.last_status_message = f"MATCHED: {cmd_event.command.value}"
            self.command_queue.put(cmd_event)
            system_logger.info(f"[VOICE INJECT] '{phrase}' -> Queue: {cmd_event.command.value}")
        else:
            self.last_status_message = self.parser.last_parse_status
        return cmd_event

    def get_pending_command(self) -> Optional[CommandEvent]:
        """Non-blocking pop of the next pending CommandEvent from the queue."""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    def _run_loop(self):
        """Main listening loop running in background."""
        while self.running:
            # 1. Ensure microphone is initialized
            if not self.is_mic_ready or self.microphone is None:
                success = self._setup_microphone()
                if not success:
                    # Sleep before retry, but keep loop alive so manual injections work
                    time.sleep(3.0)
                    continue

            # 2. Listen for audio
            try:
                self.is_listening = True
                phrase_limit = self.cfg_mgr.get("voice", "phrase_time_limit", default=6.0)
                listen_timeout = self.cfg_mgr.get("voice", "listen_timeout", default=4.0)

                with self.microphone as source:
                    # listen with timeout so thread checks self.running periodically
                    try:
                        audio = self.recognizer.listen(source, timeout=listen_timeout, phrase_time_limit=phrase_limit)
                    except sr.WaitTimeoutError:
                        continue

                # 3. Transcribe speech using offline/online recognizer
                if audio is not None and self.running:
                    try:
                        transcript = self.recognizer.recognize_google(audio)
                        self.last_transcript = transcript
                        system_logger.debug(f"[VOICE HEARD] Transcript: '{transcript}'")

                        cmd_event = self.parser.parse(transcript)
                        if cmd_event:
                            self.last_parsed_command = cmd_event.command
                            self.last_status_message = f"MATCHED: {cmd_event.command.value}"
                            self.command_queue.put(cmd_event)
                        else:
                            self.last_status_message = self.parser.last_parse_status

                    except sr.UnknownValueError:
                        # Inaudible / background noise
                        pass
                    except sr.RequestError as re:
                        system_logger.warning(f"Speech service network/request error: {re}")
                    except Exception as te:
                        system_logger.debug(f"Transcription error: {te}")

            except (IOError, OSError) as dev_err:
                self.error_count += 1
                self.is_mic_ready = False
                self.mic_status = f"DISCONNECTED ({type(dev_err).__name__})"
                system_logger.warning(f"Audio device error: {dev_err}. Retrying in 3s...")
                self.microphone = None
                time.sleep(3.0)
            except Exception as e:
                self.error_count += 1
                self.is_mic_ready = False
                self.mic_status = f"ERROR ({type(e).__name__})"
                system_logger.warning(f"Unexpected voice loop error: {e}")
                time.sleep(2.0)
            finally:
                self.is_listening = False
