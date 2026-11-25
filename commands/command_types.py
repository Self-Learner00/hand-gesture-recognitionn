from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Any
import time

class SystemMode(Enum):
    GESTURE = "GESTURE"
    VOICE = "VOICE"
    HYBRID = "HYBRID"

    @classmethod
    def from_string(cls, name: str) -> "SystemMode":
        name_clean = name.strip().upper()
        if name_clean in ("GESTURE", "GESTURES"):
            return cls.GESTURE
        elif name_clean in ("VOICE", "SPEECH"):
            return cls.VOICE
        elif name_clean in ("HYBRID", "BOTH", "ALL"):
            return cls.HYBRID
        return cls.HYBRID

class CommandType(Enum):
    NONE = "NONE"
    CONFIRM = "CONFIRM"
    CANCEL = "CANCEL"
    NEXT = "NEXT"
    PREVIOUS = "PREVIOUS"
    CLICK = "CLICK"
    RIGHT_CLICK = "RIGHT_CLICK"
    SCROLL_UP = "SCROLL_UP"
    SCROLL_DOWN = "SCROLL_DOWN"
    PAUSE = "PAUSE"
    BACK = "BACK"
    VOLUME_UP = "VOLUME_UP"
    VOLUME_DOWN = "VOLUME_DOWN"
    SWITCH_MODE = "SWITCH_MODE"

    @classmethod
    def from_string(cls, name: str) -> "CommandType":
        name_clean = name.strip().upper().replace(" ", "_")
        for cmd in cls:
            if cmd.value == name_clean or cmd.name == name_clean:
                return cmd
        return cls.NONE

@dataclass
class CommandEvent:
    command: CommandType
    source: str  # "gesture", "voice", "keyboard_shortcut", "system"
    raw_input: str  # e.g., "Thumbs Up", "confirm", "Key: M"
    confidence: float = 1.0
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()

@dataclass
class ExecutionResult:
    command: CommandType
    action_name: str
    success: bool
    message: str
    source: str
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
