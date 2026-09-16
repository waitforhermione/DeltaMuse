"""Instrument profile: loading, validation, derived pitches and key bindings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.errors import ProfileError, UnmappedPitchError
from app.core.instrument_profile import (
    DEFAULT_PROFILE_FILENAME,
    InstrumentProfile,
    KeyBinding,
    load_profile,
)

from conftest import PROFILE_DATA, make_profile


class TestLoading:
    def test_shipped_profile_is_valid(self, profile: InstrumentProfile) -> None:
        assert profile.name == "delta_harmonica"
        assert profile.base_pitch == 60
        assert profile.base_keys == ("Z", "X", "C", "V", "B", "N", "M", ",")
        assert profile.natural_intervals == (0, 2, 4, 5, 7, 9, 11, 12)
        assert profile.octave_offsets == (-1, 0, 1)
        assert profile.semitone_symbol == "#"
        assert profile.octave_up_symbol == "↑"
        assert profile.octave_down_symbol == "↓"
        assert profile.lane_count == 8

    def test_default_is_read_from_the_bundled_config(self) -> None:
        profile = InstrumentProfile.default()
        assert profile == make_profile()
        assert profile.name == "delta_harmonica"

    def test_load_from_an_explicit_path(self, tmp_path: Path) -> None:
        path = tmp_path / DEFAULT_PROFILE_FILENAME
        path.write_text(json.dumps(PROFILE_DATA, ensure_ascii=False), encoding="utf-8")
        assert InstrumentProfile.load(path) == InstrumentProfile.default()

    def test_load_profile_helper(self, tmp_path: Path) -> None:
        assert load_profile(None) == InstrumentProfile.default()

        path = tmp_path / "custom.json"
        path.write_text(json.dumps({**PROFILE_DATA, "name": "custom"}), encoding="utf-8")
        assert load_profile(path).name == "custom"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileError, match="not found"):
            InstrumentProfile.load(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(ProfileError, match="not valid JSON"):
            InstrumentProfile.load(path)

    def test_directory_instead_of_file(self, tmp_path: Path) -> None:
        with pytest.raises(ProfileError, match="not found"):
            InstrumentProfile.load(tmp_path)

    def test_non_object_json(self) -> None:
        with pytest.raises(ProfileError, match="JSON object"):
            InstrumentProfile.from_dict([1, 2, 3])  # type: ignore[arg-type]

    @pytest.mark.parametrize("field", sorted(PROFILE_DATA))
    def test_missing_required_field(self, field: str) -> None:
        data = dict(PROFILE_DATA)
        del data[field]
        with pytest.raises(ProfileError, match=f"missing required field.*{field}"):
            InstrumentProfile.from_dict(data)

    def test_describe(self, profile: InstrumentProfile) -> None:
        described = profile.describe()
        assert described["lane_count"] == 8
        assert described["pitch_min"] == 48
        assert described["pitch_max"] == 85
        assert described["binding_count"] == 48


class TestInvalidProfiles:
    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"base_keys": []}, "base_keys must not be empty"),
            ({"base_keys": ["Z", "Z", "C", "V", "B", "N", "M", ","]}, "unique"),
            ({"base_keys": ["Z", "X", "C", "V", "B", "N", "M", ""]}, "non-empty string"),
            ({"natural_intervals": [0, 2, 4]}, "lane/key count mismatch"),
            ({"natural_intervals": [0, 4, 2, 5, 7, 9, 11, 12]}, "strictly increasing"),
            ({"natural_intervals": [0, -2, 4, 5, 7, 9, 11, 12]}, "must not be negative"),
            ({"natural_intervals": [2, 4, 5, 7, 9, 11, 12, 14]}, "first natural interval must be 0"),
            ({"octave_offsets": []}, "octave_offsets must not be empty"),
            ({"octave_offsets": [-1, -1, 0]}, "octave offsets must be unique"),
            ({"octave_offsets": [-1, 1]}, "must include 0"),
            ({"base_pitch": 200}, "base_pitch must be within"),
            ({"base_pitch": "60"}, "must be an int"),
            ({"name": ""}, "name must be a non-empty string"),
            ({"semitone_symbol": ""}, "semitone_symbol must be a non-empty string"),
            ({"base_keys": "ZXCVBNM,"}, "must be a JSON array"),
            ({"natural_intervals": "0245"}, "must be a JSON array"),
        ],
    )
    def test_rejected(self, overrides: dict[str, object], match: str) -> None:
        with pytest.raises(ProfileError, match=match):
            make_profile(**overrides)

    def test_derived_pitch_outside_midi_range_is_rejected(self) -> None:
        with pytest.raises(ProfileError, match="out-of-range pitch"):
            make_profile(base_pitch=120, octave_offsets=[0, 1])

    def test_derived_pitch_below_zero_is_rejected(self) -> None:
        with pytest.raises(ProfileError, match="out-of-range pitch"):
            make_profile(base_pitch=5, octave_offsets=[0, -1])


class TestDerivedPitches:
    def test_pitches_are_derived_from_the_configuration(self) -> None:
        data = PROFILE_DATA
        expected = sorted(
            {
                int(data["base_pitch"]) + interval + 12 * offset + semitone
                for offset in [-1, 0, 1]
                for interval in data["natural_intervals"]  # type: ignore[union-attr]
                for semitone in (0, 1)
            }
        )
        assert list(make_profile().playable_pitches) == expected
        assert expected == list(range(48, 86))

    def test_shipped_range_is_forty_eight_to_eighty_five(self, profile: InstrumentProfile) -> None:
        assert profile.pitch_min == 48
        assert profile.pitch_max == 85
        assert len(profile.playable_pitches) == 38
        # The range is contiguous here, but nothing relies on that.
        assert list(profile.playable_pitches) == list(range(48, 86))

    def test_changing_base_pitch_shifts_everything(self) -> None:
        shifted = make_profile(base_pitch=62)
        assert list(shifted.playable_pitches) == list(range(50, 88))

    def test_playable_pitch_set_matches_the_tuple(self, profile: InstrumentProfile) -> None:
        assert profile.playable_pitch_set == frozenset(profile.playable_pitches)
        assert all(profile.is_playable(pitch) for pitch in profile.playable_pitches)

    def test_deterministic_ordering(self) -> None:
        first = make_profile()
        second = make_profile()
        assert first.playable_pitches == second.playable_pitches
        assert first.bindings == second.bindings

    def test_playability_uses_membership_not_a_window(self) -> None:
        """A profile with holes inside its range must be honoured."""
        holed = make_profile(base_keys=["Z", "X"], natural_intervals=[0, 12], octave_offsets=[0])

        assert holed.playable_pitches == (60, 61, 72, 73)
        assert (holed.pitch_min, holed.pitch_max) == (60, 73)
        assert holed.is_playable(72)
        assert not holed.is_playable(65)  # inside 60..73 but not playable
        assert not holed.is_playable(66)


class TestBindings:
    def test_every_binding_is_consistent(self, profile: InstrumentProfile) -> None:
        assert len(profile.bindings) == 48  # 8 lanes x 3 octaves x {natural, semitone}

        for binding in profile.bindings:
            assert isinstance(binding, KeyBinding)
            assert 0 <= binding.lane < profile.lane_count
            assert binding.key_label == profile.base_keys[binding.lane]
            assert binding.natural_interval == profile.natural_intervals[binding.lane]
            assert binding.octave_offset in profile.octave_offsets
            expected = (
                profile.base_pitch
                + binding.natural_interval
                + 12 * binding.octave_offset
                + (1 if binding.semitone else 0)
            )
            assert binding.pitch == expected

    def test_binding_order_is_deterministic(self, profile: InstrumentProfile) -> None:
        assert profile.bindings[0] == KeyBinding(0, "Z", 0, False, -1, 48)
        assert profile.bindings[-1] == KeyBinding(7, ",", 12, True, 1, 85)

    def test_bindings_for_pitch_enumerates_duplicates(self, profile: InstrumentProfile) -> None:
        bindings = profile.bindings_for_pitch(60)

        assert len(bindings) == 3
        assert [(b.lane, b.semitone, b.octave_offset) for b in bindings] == [
            (0, False, 0),  # Z, base octave
            (7, False, -1),  # ",", one octave down
            (6, True, -1),  # M#", one octave down
        ]

    def test_duplicate_representations_are_all_reported(self, profile: InstrumentProfile) -> None:
        duplicated = {
            pitch: len(profile.bindings_for_pitch(pitch))
            for pitch in profile.playable_pitches
            if len(profile.bindings_for_pitch(pitch)) > 1
        }
        assert duplicated[60] == 3  # lane 0, lane 7 one octave down, lane 6 semitone
        assert all(count >= 1 for count in duplicated.values())
        assert len(profile.bindings_for_pitch(48)) == 1
        assert len(profile.bindings_for_pitch(85)) == 1

    def test_bindings_for_unplayable_pitch_is_empty(self, profile: InstrumentProfile) -> None:
        assert profile.bindings_for_pitch(47) == ()
        assert profile.bindings_for_pitch(86) == ()
        assert profile.bindings_for_pitch(1000) == ()
        assert profile.bindings_for_pitch("not a pitch") == ()  # type: ignore[arg-type]


class TestCanonicalBinding:
    def test_prefers_the_base_octave(self, profile: InstrumentProfile) -> None:
        assert profile.canonical_binding(60).octave_offset == 0
        assert profile.canonical_binding(48).octave_offset == -1
        assert profile.canonical_binding(84).octave_offset == 1

    def test_prefers_natural_over_semitone(self, profile: InstrumentProfile) -> None:
        # 65 is a natural in the base octave; 64+1 is an alternative spelling.
        binding = profile.canonical_binding(65)
        assert binding.semitone is False
        assert binding.lane == 3

        # 66 has no natural in the base octave, so the semitone spelling wins
        # over moving a whole octave.
        binding = profile.canonical_binding(66)
        assert binding.semitone is True
        assert binding.octave_offset == 0
        assert binding.lane == 3

    def test_octave_offset_outranks_lane_order(self, profile: InstrumentProfile) -> None:
        # 72 is reachable as lane 0 one octave up and as lane 7 in the base
        # octave; staying in the base octave wins even though lane 0 is smaller.
        binding = profile.canonical_binding(72)
        assert (binding.lane, binding.octave_offset, binding.semitone) == (7, 0, False)

    def test_octave_distance_beats_lane_order(self, profile: InstrumentProfile) -> None:
        for pitch in profile.playable_pitches:
            binding = profile.canonical_binding(pitch)
            octave_distance = abs(binding.octave_offset)
            for alternative in profile.bindings_for_pitch(pitch):
                assert alternative.canonical_key >= binding.canonical_key
                assert abs(alternative.octave_offset) >= octave_distance

    def test_is_deterministic(self, profile: InstrumentProfile) -> None:
        choices = {profile.canonical_binding(72) for _ in range(20)}
        assert len(choices) == 1

    def test_round_trips_every_playable_pitch(self, profile: InstrumentProfile) -> None:
        """pitch -> canonical binding -> pitch must return the original value."""
        for pitch in profile.playable_pitches:
            binding = profile.canonical_binding(pitch)
            rebuilt = (
                profile.base_pitch
                + binding.natural_interval
                + 12 * binding.octave_offset
                + (1 if binding.semitone else 0)
            )
            assert rebuilt == pitch, f"round trip failed for {pitch}"

    def test_round_trips_every_binding(self, profile: InstrumentProfile) -> None:
        for binding in profile.bindings:
            rebuilt = (
                profile.base_pitch
                + binding.natural_interval
                + 12 * binding.octave_offset
                + (1 if binding.semitone else 0)
            )
            assert rebuilt == binding.pitch

    def test_unplayable_pitch_raises(self, profile: InstrumentProfile) -> None:
        with pytest.raises(UnmappedPitchError, match="cannot be played"):
            profile.canonical_binding(90)

    def test_binding_serialisation(self, profile: InstrumentProfile) -> None:
        described = profile.canonical_binding(60).to_dict()
        assert described == {
            "lane": 0,
            "key": "Z",
            "natural_interval": 0,
            "semitone": False,
            "octave_offset": 0,
            "pitch": 60,
        }
