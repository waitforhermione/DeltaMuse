"""Concrete audio-to-MIDI transcription backends.

Each backend is a thin adapter that hides every framework detail (torch tensors,
checkpoints, sample rates) behind :class:`app.audio.transcriber.PianoTranscriber`.
"""
