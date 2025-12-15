# Limpia tus grabaciones, normaliza y corta silencios.

import librosa
import soundfile as sf
import os
import numpy as np

RAW_DIR = "audio/raw"
CLEAN_DIR = "audio/clean"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_file(path, out, silence_gap_ms: int = 120):
    audio, sr = librosa.load(path, sr=22050)
    # Encuentra todos los segmentos no silenciosos
    intervals = librosa.effects.split(audio, top_db=47)
    # Concatenar todos los segmentos no silenciosos con pequeño silencio entre ellos
    audio_clean = []
    gap_samples = int(sr * (silence_gap_ms / 1000.0))
    gap = np.zeros(gap_samples, dtype=audio.dtype) if gap_samples > 0 else None
    for i, (start, end) in enumerate(intervals):
        audio_clean.append(audio[start:end])
        # Insertar silencio entre segmentos (no al final)
        if gap is not None and i < len(intervals) - 1:
            audio_clean.append(gap)
    if audio_clean:
        audio = librosa.util.normalize(np.concatenate(audio_clean))
    else:
        audio = librosa.util.normalize(audio)
    # soundfile solo soporta wav, flac, ogg, etc. No m4a.
    sf.write(out, audio, sr)


def clean_all_files(silence_gap_ms: int = 120):
    for file in os.listdir(RAW_DIR):
        if file.endswith(".m4a") or file.endswith(".mp3") or file.endswith(".wav"):
            # Cambia la extensión de salida a .wav si no es .wav
            print("Limpiando archivo: ", file)
            base, ext = os.path.splitext(file)
            out_file = f"{base}.wav"
            clean_file(
                os.path.join(RAW_DIR, file),
                os.path.join(CLEAN_DIR, out_file),
                silence_gap_ms=silence_gap_ms
            )
            # os.remove(os.path.join(RAW_DIR, file))
    print("Limpieza completada.")
