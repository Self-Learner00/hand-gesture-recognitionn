import json
import os
from typing import Dict, Any, Optional

DEFAULT_CONFIG: Dict[str, Any] = {
    "system": {
        "mode": "HYBRID",  # GESTURE, VOICE, HYBRID
        "dry_run": False,   # Set to True for testing without simulating real OS inputs
        "log_level": "INFO"
    },
    "gesture": {
        "min_stable_frames": 5,
        "cooldown_seconds": 0.8,
        "repeat_cooldown_seconds": 0.25,
        "confidence_threshold": 0.60,
        "require_release_to_retrigger": True,
        "hand_loss_tolerance_frames": 2,
        "mappings": {
            "1": "CONFIRM",          # Thumbs Up
            "17": "CANCEL",         # Thumbs Down
            "9": "PREVIOUS",        # Point / One Finger
            "7": "NEXT",            # Peace / Two Fingers
            "3": "PAUSE",           # Compact Palm
            "4": "BACK",            # Wave / Open Palm
            "6": "CLICK",           # OK gesture
            "18": "CLICK",          # Pinch
            "24": "RIGHT_CLICK",    # Pinched Fingers (Che Vuoi)
            "13": "SCROLL_UP",      # Count Three
            "14": "SCROLL_DOWN",    # Count Four
            "15": "VOLUME_UP",      # Call Me
            "16": "VOLUME_DOWN"     # L Shape
        },
        # Actions that are allowed to repeat without releasing hand first
        "repeatable_actions": ["SCROLL_UP", "SCROLL_DOWN", "VOLUME_UP", "VOLUME_DOWN"]
    },
    "voice": {
        "enabled": True,
        "cooldown_seconds": 0.9,
        "energy_threshold": 300,
        "dynamic_energy_threshold": True,
        "pause_threshold": 0.6,
        "phrase_time_limit": 3.0,
        "mappings": {
            "confirm": "CONFIRM",
            "enter": "CONFIRM",
            "accept": "CONFIRM",
            "yes": "CONFIRM",
            "cancel": "CANCEL",
            "escape": "CANCEL",
            "abort": "CANCEL",
            "next": "NEXT",
            "forward": "NEXT",
            "previous": "PREVIOUS",
            "prev": "PREVIOUS",
            "back": "BACK",
            "go back": "BACK",
            "pause": "PAUSE",
            "stop": "PAUSE",
            "resume": "PAUSE",
            "click": "CLICK",
            "select": "CLICK",
            "left click": "CLICK",
            "right click": "RIGHT_CLICK",
            "scroll up": "SCROLL_UP",
            "scroll down": "SCROLL_DOWN",
            "up": "SCROLL_UP",
            "down": "SCROLL_DOWN",
            "volume up": "VOLUME_UP",
            "louder": "VOLUME_UP",
            "volume down": "VOLUME_DOWN",
            "quieter": "VOLUME_DOWN",
            "switch mode": "SWITCH_MODE",
            "mode": "SWITCH_MODE",
            "gesture mode": "SWITCH_MODE",
            "voice mode": "SWITCH_MODE",
            "hybrid mode": "SWITCH_MODE"
        }
    },
    "hardware": {
        "camera_index": 0,
        "camera_width": 640,
        "camera_height": 480,
        "camera_fps": 30
    }
}

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "config", "default_config.json")
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.load()

    def load(self) -> Dict[str, Any]:
        """Loads configuration from JSON file or writes default if missing."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_cfg = json.load(f)
                self.config = self._deep_merge(DEFAULT_CONFIG.copy(), user_cfg)
                return self.config
            except Exception as e:
                print(f"[Config] Error reading {self.config_path} ({e}), using defaults.")
        
        self.config = json.loads(json.dumps(DEFAULT_CONFIG))
        self.save()
        return self.config

    def save(self):
        """Saves current configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"[Config] Failed to save config to {self.config_path}: {e}")

    def get(self, *keys: str, default: Any = None) -> Any:
        """Retrieves a nested key, e.g. cfg.get('gesture', 'min_stable_frames')."""
        curr = self.config
        for k in keys:
            if isinstance(curr, dict) and k in curr:
                curr = curr[k]
            else:
                return default
        return curr

    def set(self, *keys_and_value: Any):
        """Sets a nested key, e.g. cfg.set('system', 'mode', 'VOICE')."""
        if len(keys_and_value) < 2:
            return
        keys = keys_and_value[:-1]
        val = keys_and_value[-1]
        curr = self.config
        for k in keys[:-1]:
            if k not in curr or not isinstance(curr[k], dict):
                curr[k] = {}
            curr = curr[k]
        curr[keys[-1]] = val

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        for k, v in override.items():
            if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                base[k] = self._deep_merge(base[k], v)
            else:
                base[k] = v
        return base

# Global configuration instance
config_manager = ConfigManager()
