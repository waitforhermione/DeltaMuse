"""Key mapping: canonical bindings for every playable pitch."""

from __future__ import annotations

import pytest

from app.core.errors import UnmappedPitchError
from app.core.mapper import map_note, map_notes
from app.core.models import MelodyNote

from conftest import make_note


def melody_note(pitch: int, source_pitch: int | None = None) -> MelodyNote:
    return MelodyNote(
        source_pitch=pitch if source_pitch is None else source_pitch,
        pitch=pitch,
        start=1.5,
        duration=0.25,
        velocity=90,
        confidence=0.5,
    )


class TestMapNote:
    def test_natural_pitch_in_the_base_octave(self, profile) -> None:
        mapped = map_note(melody_note(60), profile)

        assert mapped.pitch == 60
        assert mapped.lane == 0
        assert mapped.key_label == "Z"
        assert mapped.semitone is False
        assert mapped.octave_offset == 0
        assert mapped.is_natural

    def test_semitone_pitch(self, profile) -> None:
        # 66 has no natural in the base octave, so it is played as C#+ on lane 3.
        mapped = map_note(melody_note(66), profile)

        assert mapped.pitch == 66
        assert mapped.lane == 3
        assert mapped.key_label == "V"
        assert mapped.semitone is True
        assert mapped.octave_offset == 0
        assert not mapped.is_natural

    def test_octave_down(self, profile) -> None:
        mapped = map_note(melody_note(48), profile)

        assert mapped.lane == 0
        assert mapped.key_label == "Z"
        assert mapped.semitone is False
        assert mapped.octave_offset == -1

    def test_octave_base(self, profile) -> None:
        mapped = map_note(melody_note(60), profile)
        assert mapped.octave_offset == 0

    def test_octave_up(self, profile) -> None:
        mapped = map_note(melody_note(84), profile)

        assert mapped.lane == 7
        assert mapped.key_label == ","
        assert mapped.semitone is False
        assert mapped.octave_offset == 1

    def test_provenance_is_preserved(self, profile) -> None:
        mapped = map_note(melody_note(48, source_pitch=84), profile)

        assert mapped.source_pitch == 84
        assert mapped.pitch == 48

    def test_timing_and_confidence_are_preserved(self, profile) -> None:
        source = MelodyNote(source_pitch=60, pitch=60, start=2.5, duration=1.25, velocity=70, confidence=0.25)
        mapped = map_note(source, profile)

        assert mapped.start == 2.5
        assert mapped.duration == 1.25
        assert mapped.confidence == 0.25

    def test_duplicate_representations_pick_the_canonical_one(self, profile) -> None:
        assert len(profile.bindings_for_pitch(60)) == 3
        mapped = map_note(melody_note(60), profile)

        assert (mapped.lane, mapped.semitone, mapped.octave_offset) == (0, False, 0)

    def test_mapping_is_deterministic(self, profile) -> None:
        mapped = [map_note(melody_note(72), profile) for _ in range(10)]
        assert all(note == mapped[0] for note in mapped)

    def test_invalid_pitch_raises_with_a_clear_message(self, profile) -> None:
        with pytest.raises(UnmappedPitchError, match="cannot be played on profile 'delta_harmonica'"):
            map_note(melody_note(46), profile)

    def test_pitch_above_the_range_raises(self, profile) -> None:
        with pytest.raises(UnmappedPitchError, match="playable range 48..85"):
            map_note(melody_note(90), profile)


class TestMapNotes:
    def test_maps_every_playable_pitch(self, profile) -> None:
        """Exhaustive: the full playable range must map and round-trip."""
        notes = [melody_note(pitch) for pitch in profile.playable_pitches]
        mapped, stats = map_notes(notes, profile)

        assert len(mapped) == len(profile.playable_pitches)
        assert [note.pitch for note in mapped] == list(profile.playable_pitches)

        for source, note in zip(notes, mapped):
            binding = profile.canonical_binding(note.pitch)
            assert note.lane == binding.lane
            assert note.key_label == binding.key_label == profile.base_keys[binding.lane]
            assert note.semitone == binding.semitone
            assert note.octave_offset == binding.octave_offset
            assert note.source_pitch == source.source_pitch == note.pitch

            # Round trip: rebuild the pitch from the binding.
            rebuilt = (
                profile.base_pitch
                + profile.natural_intervals[note.lane]
                + 12 * note.octave_offset
                + (1 if note.semitone else 0)
            )
            assert rebuilt == note.pitch

    def test_natural_semitone_and_octave_cases_are_all_covered(self, profile) -> None:
        notes = [melody_note(pitch) for pitch in (60, 66, 48, 84)]
        mapped, stats = map_notes(notes, profile)

        assert [note.is_natural for note in mapped] == [True, False, False, False]
        assert [note.semitone for note in mapped] == [False, True, False, False]
        assert [note.octave_offset for note in mapped] == [0, 0, -1, 1]

        assert stats.natural_count == 1
        assert stats.semitone_count == 1
        assert stats.octave_shift_count == 2
        assert stats.modifier_count == 3

    def test_empty_input(self, profile) -> None:
        mapped, stats = map_notes([], profile)

        assert mapped == ()
        assert stats.modifier_count == 0
        assert stats.natural_count == 0

    def test_mapper_does_not_transpose_or_fit(self, profile) -> None:
        """The mapper must refuse a pitch that range fitting should have fixed."""
        with pytest.raises(UnmappedPitchError):
            map_notes([melody_note(40)], profile)

    def test_deterministic_across_runs(self, profile) -> None:
        notes = [melody_note(pitch) for pitch in (48, 60, 66, 72, 84, 85)]
        first, first_stats = map_notes(notes, profile)
        second, second_stats = map_notes(notes, profile)

        assert first == second
        assert first_stats == second_stats

    def test_accepts_plain_note_events_via_duck_typing(self, profile) -> None:
        """The mapper only needs the MelodyNote surface, so notes convert cleanly."""
        from app.core.models import MelodyNote as Note

        note = Note.from_note_event(make_note(72, start=0.0, duration=0.5))
        mapped, _stats = map_notes([note], profile)
        assert mapped[0].pitch == 72
