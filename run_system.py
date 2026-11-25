#!/usr/bin/env python3
"""
Unified Hand Gesture + Voice Computer-Control System
Entry point script supporting all 3 operational modes (GESTURE, VOICE, HYBRID).
"""
import argparse
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Auto re-exec using local venv if system python is missing dependencies
venv_dir = os.path.join(script_dir, "venv")
venv_python = os.path.join(venv_dir, "bin", "python")
if os.path.exists(venv_python) and sys.prefix != venv_dir:
    try:
        import cv2
        import mediapipe
        import sklearn
    except ImportError:
        os.execv(venv_python, [venv_python] + sys.argv)

from commands.command_types import SystemMode
from system_controller import UnifiedHCIController

def get_droidcam_index() -> int:
    """Auto-detects the DroidCam v4l2 loopback device index."""
    import glob
    for name_file in sorted(glob.glob('/sys/class/video4linux/video*/name')):
        try:
            with open(name_file, 'r') as f:
                name = f.read().strip().lower()
                dev_dir = os.path.dirname(name_file)
                idx_str = os.path.basename(dev_dir).replace('video', '')
                if any(k in name for k in ['droidcam', 'dummy', 'loopback']):
                    return int(idx_str)
        except Exception:
            pass
    return 2

def main():
    parser = argparse.ArgumentParser(
        description="Unified Hand + Voice Computer-Control System (Multimodal Accessibility HCI)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["GESTURE", "VOICE", "HYBRID"],
        default="HYBRID",
        help="Input mode: GESTURE (hands only), VOICE (speech only), or HYBRID (hands + voice)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log actions without sending real OS key/mouse events (useful for testing & headless environments)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="0",
        help="Camera device index (default: 0) or 'droidcam'"
    )
    parser.add_argument(
        "--droidcam",
        action="store_true",
        help="Use DroidCam video stream (auto-detects /dev/video* loopback, default index 2)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config JSON file"
    )
    parser.add_argument(
        "--mic",
        type=str,
        default=None,
        help="Microphone device: 'earbuds', 'bluetooth', 'default', or device index integer (e.g. 24)"
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List all audio input devices / microphones and exit"
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start application in fullscreen mode (toggle with [F] key)"
    )

    args = parser.parse_args()

    if args.list_mics:
        try:
            import speech_recognition as sr
            print("Available audio input devices / microphones:")
            for i, name in enumerate(sr.Microphone.list_microphone_names()):
                is_bt = "bluez" in name.lower() or "bluetooth" in name.lower() or "headset" in name.lower()
                tag = " [BLUETOOTH / EARBUDS]" if is_bt else ""
                print(f"  [{i}] {name}{tag}")
        except Exception as e:
            print(f"Error listing microphones: {e}")
        return

    selected_mode = SystemMode.from_string(args.mode)

    if args.droidcam or str(args.camera).lower() in ("droidcam", "phone", "droid"):
        cam_idx = get_droidcam_index()
        cam_label = f"DroidCam (index {cam_idx})"
    else:
        try:
            cam_idx = int(args.camera)
            cam_label = str(cam_idx)
        except ValueError:
            cam_idx = 0
            cam_label = "0 (default fallback)"

    print("=" * 70)
    print(" UNIFIED HAND + VOICE COMPUTER CONTROL SYSTEM")
    print(f" Initial Mode: {selected_mode.value}")
    print(f" Dry Run Mode: {args.dry_run}")
    print(f" Camera Input: {cam_label}")
    if args.mic:
        print(f" Mic Device:   {args.mic}")
    print("=" * 70)
    print("Key controls:")
    print("  [M]     - Switch between GESTURE, VOICE, and HYBRID modes")
    print("  [P]     - Toggle system pause / input lock")
    print("  [T]     - Inject test voice command")
    print("  [H / Q] - Exit application")
    print("=" * 70)

    from utils.config import config_manager
    if args.mic:
        if args.mic.lower() in ("earbuds", "bluetooth", "buds", "headset"):
            config_manager.set("voice", "prefer_earbuds", True)
            config_manager.set("voice", "device_index", None)
        elif args.mic.lower() in ("default", "internal", "laptop"):
            config_manager.set("voice", "prefer_earbuds", False)
            config_manager.set("voice", "device_index", None)
        else:
            try:
                idx = int(args.mic)
                config_manager.set("voice", "device_index", idx)
            except ValueError:
                pass

    controller = UnifiedHCIController(
        config_path=args.config,
        mode=selected_mode,
        dry_run=args.dry_run
    )
    controller.run(camera_index=cam_idx, fullscreen=args.fullscreen)

if __name__ == "__main__":
    main()
