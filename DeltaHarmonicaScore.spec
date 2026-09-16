# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec: onedir, windowed (no console), bundled model + Qt.

Build with::

    .venv\\Scripts\\pyinstaller.exe DeltaHarmonicaScore.spec --noconfirm
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

datas = []
datas += collect_data_files("transkun")          # pretrained/2.0.pt + 2.0.conf
datas += collect_data_files("soundfile")         # libsndfile DLLs
datas += collect_data_files("moduleconf")
datas += [("app/config/delta_harmonica.json", "app/config")]  # shipped profile
datas += collect_dynamic_libs("torch")           # torch CPU DLLs
datas += collect_dynamic_libs("torchaudio")

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "app.audio.backends",
        "app.audio.backends.transkun_backend",
        "app.gui.main_window",
        "app.gui.worker",
        "transkun",
        "transkun.transcribe",
        "transkun.ModelTransformer",
        "moduleconf",
        "ncls",
        "soundfile",
        "pretty_midi",
        "mido",
        "soxr",
        "librosa",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "pytest",
    ],
    # TorchScript (baked into the TransKun checkpoint) needs inspect.getsource()
    # for its functions, so transkun must ship as real .py files, not inside PYZ.
    module_collection_mode={"transkun": "py", "moduleconf": "py"},
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="DeltaHarmonicaScore",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app: no console window
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="DeltaHarmonicaScore",
)
