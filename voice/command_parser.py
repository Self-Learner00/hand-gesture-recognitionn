import re
import time
from typing import Optional, Dict
from commands.command_types import CommandType, CommandEvent
from commands.command_mapping import CommandMapping
from utils.config import ConfigManager, config_manager
from utils.logger import system_logger

class VoiceCommandParser:
    """
    Parses speech-to-text transcripts into standardized CommandType events.
    Applies normalization, alias resolution, and duplicate voice suppression.
    """
    def __init__(self, command_mapping: CommandMapping, cfg_mgr: Optional[ConfigManager] = None):
        self.command_mapping = command_mapping
        self.cfg_mgr = cfg_mgr or config_manager
        self.cooldown_seconds = self.cfg_mgr.get("voice", "cooldown_seconds", default=0.35)

        self.last_phrase: str = ""
        self.last_trigger_time: float = 0.0
        self.last_parse_status: str = "Ready"

    def clean_text(self, text: str) -> str:
        """Lowercases, strips punctuation and extraneous whitespaces."""
        if not text:
            return ""
        # Remove non-alphanumeric except spaces
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return " ".join(cleaned.split())

    def parse(self, raw_transcript: str) -> Optional[CommandEvent]:
        """
        Parses a spoken transcript.
        Returns a CommandEvent if valid and not a rapid duplicate, else None.
        """
        cleaned = self.clean_text(raw_transcript)
        if not cleaned:
            self.last_parse_status = "EMPTY (No audible words detected)"
            return None

        now = time.time()

        # Check duplicate phrase suppression
        if cleaned == self.last_phrase and (now - self.last_trigger_time) < self.cooldown_seconds:
            rem = self.cooldown_seconds - (now - self.last_trigger_time)
            system_logger.debug(f"[VOICE DUPLICATE] Suppressed repeated phrase: '{cleaned}' ({rem:.1f}s remaining)")
            self.last_parse_status = f"COOLDOWN ({rem:.1f}s left for duplicate phrase)"
            return None

        cmd = self.command_mapping.get_command_for_voice(cleaned)
        if cmd == CommandType.NONE:
            system_logger.info(f"[VOICE UNMAPPED] Heard phrase '{raw_transcript}' (cleaned: '{cleaned}') - no matching command.")
            self.last_parse_status = f"UNMAPPED (No action for '{cleaned[:22]}')"
            return None

        self.last_phrase = cleaned
        self.last_trigger_time = now
        self.last_parse_status = f"MATCHED: {cmd.value}"

        system_logger.info(
            f"[VOICE TRIGGER] Heard: '{raw_transcript}' -> COMMAND: {cmd.value}"
        )

        return CommandEvent(
            command=cmd,
            source="voice",
            raw_input=raw_transcript,
            confidence=0.95,
            timestamp=now
        )
