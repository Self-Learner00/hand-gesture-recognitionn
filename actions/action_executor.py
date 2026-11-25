import time
from typing import Optional, Callable
from commands.command_types import CommandType, CommandEvent, ExecutionResult, SystemMode
from actions.keyboard_actions import KeyboardActionHandler
from actions.mouse_actions import MouseActionHandler
from actions.system_actions import SystemActionHandler
from utils.config import ConfigManager, config_manager
from utils.logger import system_logger

class ActionExecutor:
    """
    Centralized executor for all computer actions triggered by either Hand Gestures or Voice.
    Guarantees both input modalities go through the identical execution and debouncing pipeline.
    """
    def __init__(self, cfg_mgr: Optional[ConfigManager] = None, on_mode_change: Optional[Callable[[SystemMode], None]] = None):
        self.cfg_mgr = cfg_mgr or config_manager
        self.on_mode_change = on_mode_change

        dry_run = self.cfg_mgr.get("system", "dry_run", default=False)
        self.keyboard = KeyboardActionHandler(dry_run=dry_run)
        self.mouse = MouseActionHandler(dry_run=dry_run)
        self.system = SystemActionHandler(dry_run=dry_run)

        # State tracking
        self.last_command: CommandType = CommandType.NONE
        self.last_action_name: str = "IDLE"
        self.last_source: str = "NONE"
        self.last_status: str = "READY"
        self.last_execution_time: float = 0.0
        self.active_cooldown: float = 0.0

    @property
    def is_paused(self) -> bool:
        return self.system.is_paused

    def cooldown_remaining(self) -> float:
        """Returns seconds remaining in cooldown, or 0.0 if ready."""
        elapsed = time.time() - self.last_execution_time
        rem = self.active_cooldown - elapsed
        return max(0.0, rem)

    def is_cooling_down(self) -> bool:
        return self.cooldown_remaining() > 0.0

    def execute(self, event: CommandEvent, current_mode: SystemMode) -> ExecutionResult:
        """
        Executes an action for the given CommandEvent, checking mode constraints and system pause state.
        """
        now = time.time()

        # 1. Validate Mode
        if event.source == "gesture" and current_mode == SystemMode.VOICE:
            return ExecutionResult(
                command=event.command,
                action_name="IGNORED_MODE",
                success=False,
                message=f"Gesture ignored in VOICE mode",
                source=event.source
            )

        if event.source == "voice" and current_mode == SystemMode.GESTURE:
            return ExecutionResult(
                command=event.command,
                action_name="IGNORED_MODE",
                success=False,
                message=f"Voice ignored in GESTURE mode",
                source=event.source
            )

        # 2. Check Pause State
        if self.is_paused and event.command != CommandType.PAUSE and event.command != CommandType.SWITCH_MODE:
            self.last_status = "SYSTEM_PAUSED"
            return ExecutionResult(
                command=event.command,
                action_name="PAUSED",
                success=False,
                message="System is paused. Gesture/Voice inputs locked.",
                source=event.source
            )

        # 3. Check Cooldown
        if self.is_cooling_down():
            self.last_status = f"COOLDOWN ({self.cooldown_remaining():.1f}s)"
            return ExecutionResult(
                command=event.command,
                action_name="COOLDOWN",
                success=False,
                message=f"Ignored due to cooldown: {self.cooldown_remaining():.2f}s remaining",
                source=event.source
            )

        # Determine cooldown for this action
        if event.command in (CommandType.SCROLL_UP, CommandType.SCROLL_DOWN, CommandType.VOLUME_UP, CommandType.VOLUME_DOWN):
            cooldown = self.cfg_mgr.get("gesture", "repeat_cooldown_seconds", default=0.25)
        elif event.source == "voice":
            cooldown = self.cfg_mgr.get("voice", "cooldown_seconds", default=0.35)
        else:
            cooldown = self.cfg_mgr.get("gesture", "cooldown_seconds", default=0.8)

        # 4. Dispatch Action
        action_name = "NONE"
        try:
            if event.command == CommandType.CONFIRM:
                action_name = self.keyboard.confirm()
            elif event.command == CommandType.CANCEL:
                action_name = self.keyboard.cancel()
            elif event.command == CommandType.NEXT:
                action_name = self.keyboard.next()
            elif event.command == CommandType.PREVIOUS:
                action_name = self.keyboard.previous()
            elif event.command == CommandType.CLICK:
                action_name = self.mouse.click("left")
            elif event.command == CommandType.RIGHT_CLICK:
                action_name = self.mouse.click("right")
            elif event.command == CommandType.SCROLL_UP:
                action_name = self.mouse.scroll(1)
            elif event.command == CommandType.SCROLL_DOWN:
                action_name = self.mouse.scroll(-1)
            elif event.command == CommandType.BACK:
                action_name = self.keyboard.back()
            elif event.command == CommandType.VOLUME_UP:
                action_name = self.system.change_volume(+5)
            elif event.command == CommandType.VOLUME_DOWN:
                action_name = self.system.change_volume(-5)
            elif event.command == CommandType.PAUSE:
                action_name = self.system.toggle_pause()
            elif event.command == CommandType.SWITCH_MODE:
                new_mode = self.system.cycle_mode(current_mode)
                action_name = f"MODE_{new_mode.value}"
                if self.on_mode_change:
                    self.on_mode_change(new_mode)
            else:
                return ExecutionResult(
                    command=event.command,
                    action_name="UNKNOWN",
                    success=False,
                    message=f"No action handler for command {event.command}",
                    source=event.source
                )

            # Update State
            self.last_command = event.command
            self.last_action_name = action_name
            self.last_source = event.source.upper()
            self.last_status = "EXECUTED"
            self.last_execution_time = now
            self.active_cooldown = cooldown

            system_logger.info(
                f"[EXECUTE SUCCESS] Source: {event.source.upper()} | "
                f"Command: {event.command.value} | Action: {action_name}"
            )

            return ExecutionResult(
                command=event.command,
                action_name=action_name,
                success=True,
                message=f"Action '{action_name}' executed successfully",
                source=event.source,
                timestamp=now
            )

        except Exception as e:
            system_logger.error(f"[EXECUTE ERROR] Command {event.command.value} failed: {e}", exc_info=True)
            self.last_status = f"ERROR: {e}"
            return ExecutionResult(
                command=event.command,
                action_name=action_name,
                success=False,
                message=f"Execution error: {e}",
                source=event.source,
                timestamp=now
            )
