import os
import speech_recognition as sr
from pydub import AudioSegment

AUDIO_OUT_DIR = "audio/transcribed"
METADATA_PATH = "data/metadata.csv"

def segmentar_audio_por_palabras(audio_path, num_palabras):
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(audio_path)
    duracion_ms = len(audio)
    paso_ms = 2000  # 2 segundos por bloque para reconocimiento, ajustable
    inicio = 0
    contador = 1
    palabras_acumuladas = []
    texto_acumulado = ""
    fragmento_inicio = 0
    ultimo_fin = 0

    while inicio < duracion_ms:
        fin = min(inicio + paso_ms, duracion_ms)
        segmento = audio[inicio:fin]
        temp_path = "temp_segment.wav"
        segmento.export(temp_path, format="wav")
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            try:
                texto = recognizer.recognize_google(audio_data, language="es-ES")
            except Exception:
                texto = ""
        os.remove(temp_path)
        if texto.strip():
            palabras = texto.strip().split()
            palabras_acumuladas.extend(palabras)
            texto_acumulado += (" " if texto_acumulado else "") + texto.strip()
            ultimo_fin = fin  # Solo avanzar si hay texto reconocido
        if len(palabras_acumuladas) >= num_palabras:
            out_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{contador}.wav"
            out_path = os.path.join(AUDIO_OUT_DIR, out_filename)
            fragmento = audio[fragmento_inicio:ultimo_fin]
            fragmento.export(out_path, format="wav")
            with open(METADATA_PATH, "a", encoding="utf-8") as meta:
                meta.write(f"{os.path.splitext(out_filename)[0]}|{texto_acumulado.strip()}\n")
            contador += 1
            palabras_acumuladas = []
            texto_acumulado = ""
            fragmento_inicio = ultimo_fin  # Avanzar el inicio al final del último bloque reconocido
        inicio = fin

    # Si quedan palabras al final, guardar el último fragmento
    if palabras_acumuladas and texto_acumulado and ultimo_fin > fragmento_inicio:
        out_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{contador}.wav"
        out_path = os.path.join(AUDIO_OUT_DIR, out_filename)
        fragmento = audio[fragmento_inicio:ultimo_fin]
        fragmento.export(out_path, format="wav")
        with open(METADATA_PATH, "a", encoding="utf-8") as meta:
            meta.write(f"{os.path.splitext(out_filename)[0]}|{texto_acumulado.strip()}\n")

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 3:
#         print("Uso: python segmentar_audio.py <archivo_wav> <num_palabras>")
#         exit(1)
#     audio_path = sys.argv[1]
#     num_palabras = int(sys.argv[2])
#     segmentar_audio_por_palabras(audio_path, num_palabras)
