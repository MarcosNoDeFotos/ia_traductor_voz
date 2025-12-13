# Limpia tus grabaciones, normaliza y corta silencios.

import librosa
import soundfile as sf
import os
import numpy as np

RAW_DIR = "audio/raw"
CLEAN_DIR = "audio/clean"

os.makedirs(CLEAN_DIR, exist_ok=True)

def clean_file(path, out):
    audio, sr = librosa.load(path, sr=22050)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    audio = librosa.util.normalize(audio)
    # soundfile solo soporta wav, flac, ogg, etc. No m4a.
    sf.write(out, audio, sr)




def clean_file_v2(path, out):
    audio, sr = librosa.load(path, sr=22050)
    # Encuentra todos los segmentos no silenciosos
    intervals = librosa.effects.split(audio, top_db=25)
    # Concatenar todos los segmentos no silenciosos
    audio_clean = []
    for start, end in intervals:
        audio_clean.append(audio[start:end])
    if audio_clean:
        audio = librosa.util.normalize(np.concatenate(audio_clean))
    else:
        audio = librosa.util.normalize(audio)
    # soundfile solo soporta wav, flac, ogg, etc. No m4a.
    sf.write(out, audio, sr)


def clean_all_files():
    for file in os.listdir(RAW_DIR):
        if file.endswith(".m4a") or file.endswith(".mp3") or file.endswith(".wav"):
            # Cambia la extensión de salida a .wav si no es .wav
            base, ext = os.path.splitext(file)
            out_file = f"{base}.wav"
            clean_file_v2(
                os.path.join(RAW_DIR, file),
                os.path.join(CLEAN_DIR, out_file)
            )
            os.remove(os.path.join(RAW_DIR, file))
    print("Limpieza completada.")
