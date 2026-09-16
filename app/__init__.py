"""Piano audio -> Delta Harmonica static score generator.

Strictly layered pipeline (layers must never reach across each other)::

    Audio -> Transcription -> NoteEvent -> Melody Processing
          -> PlayableNote -> Score Layout -> Renderer
          -> ConversionService -> CLI / Windows GUI
"""

from app.version import APP_NAME, VERSION

__version__ = VERSION
__all__ = ["__version__", "APP_NAME"]
