"""Instrument profile: which pitches an instrument can play, and how.

A profile is pure configuration (``app/config/delta_harmonica.json``). Everything
else is *derived* from it::

    pitch(lane, octave_offset, semitone)
        = base_pitch + natural_intervals[lane] + 12 * octave_offset + (1 if semitone else 0)

Nothing in this project hardcodes a pitch range such as ``48..85``: the playable
pitches are computed from the profile, so a profile with holes inside its range
works exactly the same way.

Pure Python, standard library only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.errors import ProfileError, UnmappedPitchError
from app.core.models import MAX_MIDI_PITCH, MIN_MIDI_PITCH

DEFAULT_PROFILE_FILENAME = "delta_harmonica.json"
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[1] / "config" / DEFAULT_PROFILE_FILENAME

#: Upper bound used when a pitch has to be searched across octaves.
_OCTAVE = 12

_REQUIRED_FIELDS = (
    "name",
    "base_pitch",
    "base_keys",
    "natural_intervals",
    "octave_offsets",
    "semitone_symbol",
    "octave_up_symbol",
    "octave_down_symbol",
)


@dataclass(frozen=True)
class KeyBinding:
    """One concrete way of playing a pitch on the instrument.

    A single pitch usually has several: the same note can be reached through
    different lanes, through the semitone modifier, or in a different octave.
    """

    lane: int
    key_label: str
    natural_interval: int
    semitone: bool
    octave_offset: int
    pitch: int

    @property
    def canonical_key(self) -> tuple[int, int, int]:
        """Ranking used to pick the preferred binding of a pitch.

        1. smallest ``abs(octave_offset)`` (stay in the base octave),
        2. natural before semitone,
        3. smaller lane index.
        """
        return (abs(self.octave_offset), 1 if self.semitone else 0, self.lane)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane,
            "key": self.key_label,
            "natural_interval": self.natural_interval,
            "semitone": self.semitone,
            "octave_offset": self.octave_offset,
            "pitch": self.pitch,
        }


@dataclass(frozen=True)
class InstrumentProfile:
    """A validated instrument profile with all its derived lookup tables."""

    name: str
    base_pitch: int
    base_keys: tuple[str, ...]
    natural_intervals: tuple[int, ...]
    octave_offsets: tuple[int, ...]
    semitone_symbol: str
    octave_up_symbol: str
    octave_down_symbol: str

    # -- derived ---------------------------------------------------------- #
    bindings: tuple[KeyBinding, ...] = field(init=False, repr=False, compare=False)
    playable_pitches: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _playable_pitch_set: frozenset[int] = field(init=False, repr=False, compare=False)
    _bindings_by_pitch: dict[int, tuple[KeyBinding, ...]] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validate()
        bindings = self._derive_bindings()

        grouped: dict[int, list[KeyBinding]] = {}
        for binding in bindings:
            grouped.setdefault(binding.pitch, []).append(binding)

        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "playable_pitches", tuple(sorted(grouped)))
        object.__setattr__(self, "_playable_pitch_set", frozenset(grouped))
        object.__setattr__(
            self,
            "_bindings_by_pitch",
            {
                pitch: tuple(
                    sorted(
                        entries,
                        key=lambda entry: (
                            entry.canonical_key,
                            entry.natural_interval,
                            entry.octave_offset,
                        ),
                    )
                )
                for pitch, entries in grouped.items()
            },
        )

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InstrumentProfile":
        """Build a profile from already parsed JSON data."""
        if not isinstance(data, Mapping):
            raise ProfileError(f"instrument profile must be a JSON object, got {type(data).__name__}")

        missing = [key for key in _REQUIRED_FIELDS if key not in data]
        if missing:
            raise ProfileError(f"instrument profile is missing required field(s): {', '.join(missing)}")

        for key in ("base_keys", "natural_intervals", "octave_offsets"):
            value = data[key]
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise ProfileError(f"{key} must be a JSON array, got {type(value).__name__}")

        try:
            return cls(
                name=str(data["name"]),
                base_pitch=data["base_pitch"],
                base_keys=tuple(data["base_keys"]),
                natural_intervals=tuple(data["natural_intervals"]),
                octave_offsets=tuple(data["octave_offsets"]),
                semitone_symbol=str(data["semitone_symbol"]),
                octave_up_symbol=str(data["octave_up_symbol"]),
                octave_down_symbol=str(data["octave_down_symbol"]),
            )
        except TypeError as exc:
            raise ProfileError(f"instrument profile has malformed fields: {exc}") from exc

    @classmethod
    def load(cls, path: str | Path) -> "InstrumentProfile":
        """Read and validate a profile from a JSON file."""
        resolved = Path(path)
        if not resolved.is_file():
            raise ProfileError(f"instrument profile not found: {resolved}")
        try:
            raw = resolved.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProfileError(f"cannot read instrument profile {resolved}: {exc}") from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProfileError(f"instrument profile {resolved} is not valid JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def default(cls) -> "InstrumentProfile":
        """The shipped Delta Harmonica profile.

        Resolved through the resource helper so it works both from source and
        inside a bundled EXE (never depends on the current working directory).
        """
        from app.resources import resource_path

        bundled = resource_path("app", "config", DEFAULT_PROFILE_FILENAME)
        if bundled.is_file():
            return cls.load(bundled)
        return cls.load(DEFAULT_PROFILE_PATH)

    # ------------------------------------------------------------------ #
    # queries
    # ------------------------------------------------------------------ #
    @property
    def lane_count(self) -> int:
        return len(self.base_keys)

    @property
    def playable_pitch_set(self) -> frozenset[int]:
        return self._playable_pitch_set

    @property
    def pitch_min(self) -> int:
        return self.playable_pitches[0]

    @property
    def pitch_max(self) -> int:
        return self.playable_pitches[-1]

    def is_playable(self, pitch: int) -> bool:
        """``True`` only when *pitch* is one of the profile's playable pitches.

        Membership is tested against the derived set, not against a min/max
        window, so a profile with holes inside its range behaves correctly.
        """
        return pitch in self._playable_pitch_set

    def bindings_for_pitch(self, pitch: int) -> tuple[KeyBinding, ...]:
        """Every legal way of playing *pitch* (empty when it is not playable)."""
        try:
            return self._bindings_by_pitch.get(int(pitch), ())
        except (TypeError, ValueError):
            return ()

    def canonical_binding(self, pitch: int) -> KeyBinding:
        """The preferred binding of *pitch*.

        Raises
        ------
        UnmappedPitchError
            The pitch is not playable on this instrument.
        """
        bindings = self.bindings_for_pitch(pitch)
        if not bindings:
            raise UnmappedPitchError(
                f"pitch {int(pitch)} cannot be played on profile {self.name!r} "
                f"(playable range {self.pitch_min}..{self.pitch_max}, {len(self.playable_pitches)} pitches)"
            )
        return bindings[0]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "lanes": list(self.base_keys),
            "lane_count": self.lane_count,
            "base_pitch": self.base_pitch,
            "natural_intervals": list(self.natural_intervals),
            "octave_offsets": list(self.octave_offsets),
            "pitch_min": self.pitch_min,
            "pitch_max": self.pitch_max,
            "playable_pitch_count": len(self.playable_pitches),
            "binding_count": len(self.bindings),
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProfileError(f"profile name must be a non-empty string, got {self.name!r}")

        if len(self.base_keys) == 0:
            raise ProfileError("base_keys must not be empty")
        if not all(isinstance(key, str) and key for key in self.base_keys):
            raise ProfileError(f"every base key must be a non-empty string, got {self.base_keys!r}")
        if len(set(self.base_keys)) != len(self.base_keys):
            raise ProfileError(f"base keys must be unique, got {self.base_keys!r}")

        if len(self.natural_intervals) != len(self.base_keys):
            raise ProfileError(
                f"lane/key count mismatch: {len(self.natural_intervals)} natural intervals "
                f"for {len(self.base_keys)} keys"
            )
        intervals = [self._as_int(value, "natural_interval") for value in self.natural_intervals]
        if any(value < 0 for value in intervals):
            raise ProfileError(f"natural intervals must not be negative, got {intervals!r}")
        if any(later <= earlier for earlier, later in zip(intervals, intervals[1:])):
            raise ProfileError(f"natural intervals must be strictly increasing, got {intervals!r}")
        if intervals[0] != 0:
            raise ProfileError(
                f"the first natural interval must be 0 so that base_pitch is the first key's pitch, "
                f"got {intervals[0]}"
            )

        offsets = [self._as_int(value, "octave_offset") for value in self.octave_offsets]
        if not offsets:
            raise ProfileError("octave_offsets must not be empty")
        if len(set(offsets)) != len(offsets):
            raise ProfileError(f"octave offsets must be unique, got {offsets!r}")
        if 0 not in offsets:
            raise ProfileError(f"octave_offsets must include 0, got {offsets!r}")

        base_pitch = self._as_int(self.base_pitch, "base_pitch")
        if not MIN_MIDI_PITCH <= base_pitch <= MAX_MIDI_PITCH:
            raise ProfileError(
                f"base_pitch must be within [{MIN_MIDI_PITCH}, {MAX_MIDI_PITCH}], got {base_pitch}"
            )

        for name, symbol in (
            ("semitone_symbol", self.semitone_symbol),
            ("octave_up_symbol", self.octave_up_symbol),
            ("octave_down_symbol", self.octave_down_symbol),
        ):
            if not isinstance(symbol, str) or not symbol:
                raise ProfileError(f"{name} must be a non-empty string, got {symbol!r}")

    def _derive_bindings(self) -> tuple[KeyBinding, ...]:
        """Enumerate every legal (lane, octave offset, semitone) representation.

        Order is deterministic: octave offsets ascending, then lanes, then natural
        before semitone.
        """
        bindings: list[KeyBinding] = []
        for octave_offset in sorted(int(value) for value in self.octave_offsets):
            for lane, interval in enumerate(int(value) for value in self.natural_intervals):
                natural = int(self.base_pitch) + interval + _OCTAVE * octave_offset
                for semitone in (False, True):
                    pitch = natural + (1 if semitone else 0)
                    if not MIN_MIDI_PITCH <= pitch <= MAX_MIDI_PITCH:
                        raise ProfileError(
                            f"profile {self.name!r} derives out-of-range pitch {pitch} "
                            f"(lane {lane}, octave offset {octave_offset}, semitone={semitone})"
                        )
                    bindings.append(
                        KeyBinding(
                            lane=lane,
                            key_label=self.base_keys[lane],
                            natural_interval=interval,
                            semitone=semitone,
                            octave_offset=octave_offset,
                            pitch=pitch,
                        )
                    )
        return tuple(bindings)

    @staticmethod
    def _as_int(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProfileError(f"{name} must be an int, got {value!r}")
        return value


def load_profile(path: str | Path | None = None) -> InstrumentProfile:
    """Load a profile from *path*, or the shipped default when path is ``None``."""
    return InstrumentProfile.default() if path is None else InstrumentProfile.load(path)


__all__ = [
    "DEFAULT_PROFILE_FILENAME",
    "DEFAULT_PROFILE_PATH",
    "KeyBinding",
    "InstrumentProfile",
    "load_profile",
]
