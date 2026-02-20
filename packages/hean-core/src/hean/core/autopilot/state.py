"""AutoPilot state machine — manages mode transitions with hysteresis."""

from __future__ import annotations

import time

from hean.logging import get_logger

from .types import AutoPilotMode

logger = get_logger(__name__)

# Minimum time in a state before allowing transition (prevents oscillation)
_MIN_STATE_DURATION_SEC: dict[AutoPilotMode, float] = {
    AutoPilotMode.LEARNING: 0.0,  # Can exit anytime (timer-based)
    AutoPilotMode.CONSERVATIVE: 300.0,  # 5 min minimum
    AutoPilotMode.BALANCED: 120.0,  # 2 min minimum
    AutoPilotMode.AGGRESSIVE: 60.0,  # 1 min (can exit fast on danger)
    AutoPilotMode.PROTECTIVE: 600.0,  # 10 min minimum (don't oscillate)
    AutoPilotMode.EVOLVING: 0.0,  # Controlled by evolution cycle
}

# Valid transitions (from -> set of allowed targets)
_VALID_TRANSITIONS: dict[AutoPilotMode, set[AutoPilotMode]] = {
    AutoPilotMode.LEARNING: {AutoPilotMode.CONSERVATIVE},
    AutoPilotMode.CONSERVATIVE: {
        AutoPilotMode.BALANCED,
        AutoPilotMode.PROTECTIVE,
        AutoPilotMode.EVOLVING,
    },
    AutoPilotMode.BALANCED: {
        AutoPilotMode.AGGRESSIVE,
        AutoPilotMode.CONSERVATIVE,
        AutoPilotMode.PROTECTIVE,
        AutoPilotMode.EVOLVING,
    },
    AutoPilotMode.AGGRESSIVE: {
        AutoPilotMode.BALANCED,
        AutoPilotMode.PROTECTIVE,
        AutoPilotMode.EVOLVING,
    },
    AutoPilotMode.PROTECTIVE: {
        AutoPilotMode.CONSERVATIVE,
        AutoPilotMode.EVOLVING,
    },
    AutoPilotMode.EVOLVING: {
        # Can return to any operational state
        AutoPilotMode.CONSERVATIVE,
        AutoPilotMode.BALANCED,
        AutoPilotMode.AGGRESSIVE,
        AutoPilotMode.PROTECTIVE,
    },
}


class AutoPilotStateMachine:
    """Finite state machine for AutoPilot operating modes.

    Enforces valid transitions, minimum state durations (hysteresis),
    and tracks transition history.
    """

    def __init__(self, initial_mode: AutoPilotMode = AutoPilotMode.LEARNING) -> None:
        self._mode = initial_mode
        self._previous_mode: AutoPilotMode | None = None
        self._mode_entered_at: float = time.monotonic()
        self._transition_count = 0
        self._history: list[tuple[float, AutoPilotMode, AutoPilotMode, str]] = []

        # State saved before entering EVOLVING (to restore after)
        self._pre_evolving_mode: AutoPilotMode | None = None

    @property
    def mode(self) -> AutoPilotMode:
        """Current operating mode."""
        return self._mode

    @property
    def previous_mode(self) -> AutoPilotMode | None:
        """Previous operating mode."""
        return self._previous_mode

    @property
    def time_in_current_mode(self) -> float:
        """Seconds spent in current mode."""
        return time.monotonic() - self._mode_entered_at

    @property
    def transition_count(self) -> int:
        """Total number of transitions."""
        return self._transition_count

    def can_transition(self, target: AutoPilotMode) -> bool:
        """Check if a transition to target mode is allowed right now."""
        if target == self._mode:
            return False

        # Check valid transition
        valid = _VALID_TRANSITIONS.get(self._mode, set())
        if target not in valid:
            return False

        # Check minimum duration (hysteresis)
        min_duration = _MIN_STATE_DURATION_SEC.get(self._mode, 0.0)
        if self.time_in_current_mode < min_duration:
            return False

        return True

    def transition(self, target: AutoPilotMode, reason: str = "") -> bool:
        """Attempt a state transition.

        Args:
            target: Target mode.
            reason: Human-readable reason for the transition.

        Returns:
            True if the transition was successful.
        """
        if not self.can_transition(target):
            logger.debug(
                "[AutoPilot] Transition %s -> %s denied (min_duration=%.0fs, "
                "elapsed=%.0fs)",
                self._mode.value,
                target.value,
                _MIN_STATE_DURATION_SEC.get(self._mode, 0.0),
                self.time_in_current_mode,
            )
            return False

        old_mode = self._mode
        self._previous_mode = old_mode
        self._mode = target
        self._mode_entered_at = time.monotonic()
        self._transition_count += 1

        # Save pre-evolving state
        if target == AutoPilotMode.EVOLVING:
            self._pre_evolving_mode = old_mode

        self._history.append((time.time(), old_mode, target, reason))

        # Keep history bounded
        if len(self._history) > 500:
            self._history = self._history[-250:]

        logger.info(
            "[AutoPilot] Mode transition: %s -> %s (reason: %s)",
            old_mode.value,
            target.value,
            reason or "unspecified",
        )
        return True

    def exit_evolving(self) -> AutoPilotMode:
        """Return to the mode that was active before EVOLVING.

        If current mode is not EVOLVING, does nothing and returns current mode.
        """
        if self._mode != AutoPilotMode.EVOLVING:
            return self._mode

        restore_to = self._pre_evolving_mode or AutoPilotMode.BALANCED
        self.transition(restore_to, reason="evolution_cycle_complete")
        return self._mode

    def force_protective(self, reason: str = "safety_trigger") -> None:
        """Force transition to PROTECTIVE regardless of hysteresis.

        This is a safety escape hatch — bypasses minimum duration checks.
        Used when KillSwitch or extreme conditions are detected.
        """
        if self._mode == AutoPilotMode.PROTECTIVE:
            return

        old_mode = self._mode
        self._previous_mode = old_mode
        self._mode = AutoPilotMode.PROTECTIVE
        self._mode_entered_at = time.monotonic()
        self._transition_count += 1
        self._history.append((time.time(), old_mode, AutoPilotMode.PROTECTIVE, reason))

        logger.warning(
            "[AutoPilot] FORCED PROTECTIVE: %s -> PROTECTIVE (reason: %s)",
            old_mode.value,
            reason,
        )

    def get_status(self) -> dict:
        """Get current state machine status."""
        return {
            "mode": self._mode.value,
            "previous_mode": self._previous_mode.value if self._previous_mode else None,
            "time_in_mode_sec": round(self.time_in_current_mode, 1),
            "transition_count": self._transition_count,
            "recent_transitions": [
                {
                    "timestamp": t[0],
                    "from": t[1].value,
                    "to": t[2].value,
                    "reason": t[3],
                }
                for t in self._history[-10:]
            ],
        }
