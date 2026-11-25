import re
from typing import Optional, Dict
from commands.command_types import CommandType
from utils.config import ConfigManager, config_manager
from utils.logger import system_logger

class CommandMapping:
    """
    Translates detected gestures and parsed voice phrases into unified CommandTypes.
    Configurable at runtime through ConfigManager.
    """
    def __init__(self, cfg_mgr: Optional[ConfigManager] = None):
        self.cfg_mgr = cfg_mgr or config_manager
        self.gesture_map: Dict[int, CommandType] = {}
        self.voice_map: Dict[str, CommandType] = {}
        self.reload_mappings()

    def reload_mappings(self):
        """Reloads mappings from configuration."""
        raw_gesture_map = self.cfg_mgr.get("gesture", "mappings", default={})
        self.gesture_map.clear()
        for gid_str, cmd_name in raw_gesture_map.items():
            try:
                gid = int(gid_str)
                cmd = CommandType.from_string(cmd_name)
                if cmd != CommandType.NONE:
                    self.gesture_map[gid] = cmd
            except (ValueError, TypeError):
                continue

        raw_voice_map = self.cfg_mgr.get("voice", "mappings", default={})
        self.voice_map.clear()
        for phrase, cmd_name in raw_voice_map.items():
            clean_phrase = phrase.strip().lower()
            cmd = CommandType.from_string(cmd_name)
            if cmd != CommandType.NONE:
                self.voice_map[clean_phrase] = cmd

        system_logger.info(
            f"Command mappings loaded: {len(self.gesture_map)} gestures, {len(self.voice_map)} voice phrases."
        )

    def get_command_for_gesture(self, gesture_id: int) -> CommandType:
        """Returns CommandType for a given gesture ID or CommandType.NONE if unmapped."""
        return self.gesture_map.get(gesture_id, CommandType.NONE)

    def get_command_for_voice(self, phrase: str) -> CommandType:
        """Returns CommandType for a given spoken phrase or CommandType.NONE."""
        clean_phrase = phrase.strip().lower()
        if not clean_phrase:
            return CommandType.NONE

        if clean_phrase in self.voice_map:
            return self.voice_map[clean_phrase]

        # Match longest phrases first to prioritize specific phrases over single words (e.g. "volume up" over "up")
        sorted_mappings = sorted(self.voice_map.items(), key=lambda item: len(item[0]), reverse=True)
        for mapped_phrase, cmd in sorted_mappings:
            # Word boundary matching in sentence (e.g. "please confirm this" matches "confirm")
            pattern = r'\b' + re.escape(mapped_phrase) + r'\b'
            if re.search(pattern, clean_phrase):
                return cmd

        return CommandType.NONE

    def set_gesture_mapping(self, gesture_id: int, command: CommandType):
        self.gesture_map[gesture_id] = command
        self.cfg_mgr.set("gesture", "mappings", str(gesture_id), command.value)
        self.cfg_mgr.save()

    def set_voice_mapping(self, phrase: str, command: CommandType):
        clean = phrase.strip().lower()
        self.voice_map[clean] = command
        self.cfg_mgr.set("voice", "mappings", clean, command.value)
        self.cfg_mgr.save()

    def is_repeatable_gesture_command(self, cmd: CommandType) -> bool:
        repeatables = self.cfg_mgr.get("gesture", "repeatable_actions", default=[])
        return cmd.value in repeatables
