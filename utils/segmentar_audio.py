import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent
from tqdm import tqdm
from time import sleep
METADATA_PATH = "data/metadata.csv"
SPLITTED_DIR = "audio/splitted"

# Parámetros de silencio (ajustables)
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -30

def _trim_leading_trailing_silence(seg: AudioSegment) -> AudioSegment:
    # Detecta regiones no silenciosas y recorta a ese rango
    ranges = detect_nonsilent(seg, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESH_DBFS)
    if not ranges:
        return AudioSegment.silent(duration=0, frame_rate=seg.frame_rate)
    start, end = ranges[0][0], ranges[-1][1]
    return seg[start:end]

def _trim_trailing_silence_only(seg: AudioSegment) -> AudioSegment:
    # Recorta solo el silencio al final, mantiene el inicio intacto
    ranges = detect_nonsilent(seg, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESH_DBFS)
    if not ranges:
        return AudioSegment.silent(duration=0, frame_rate=seg.frame_rate)
    end = ranges[-1][1]
    return seg[:end]

def segmentar_audio_por_palabras(
    audio_path,
    min_segment_ms: int = 8000,
    min_silence_ms: int = 100,
    silence_thresh_dbfs: int = SILENCE_THRESH_DBFS,
    maximo_cortes=None,
    tail_after_silence_ms: int = 200,  # ms extra tras el silencio para evitar cortar palabras
    tail_before_silence_ms: int = 200  # ms extra antes del inicio para evitar cortar palabras
):
    # Exporta segmentos limpios a SPLITTED_DIR (sin transcribir)
    os.makedirs(SPLITTED_DIR, exist_ok=True)
    recognizer = sr.Recognizer()  # ...existing code uses sr for temp I/O; kept for consistency though not used now
    audio = AudioSegment.from_wav(audio_path)
    duracion_ms = len(audio)

    # Primer pasada: dividir en segmentos con mínimo min_segment_ms y cortar en el primer silencio >= min_silence_ms
    segmentos = []
    cursor = 0
    print("Segmentando audio...")
    sleep(0.2)
    # Barra de progreso para segmentación (consume en ms)
    if maximo_cortes:
        pbar = tqdm(total=maximo_cortes, unit="pasos", desc="Segmentando", leave=False)
    else:
        pbar = tqdm(total=duracion_ms, unit="ms", desc="Segmentando", leave=False)
    while cursor < duracion_ms and (maximo_cortes is None or len(segmentos) < maximo_cortes):
        start = cursor
        
        # Garantizar mínimo
        after_min = min(start + min_segment_ms, duracion_ms)
        if after_min >= duracion_ms:
            segmentos.append(audio[start:duracion_ms])
            if not maximo_cortes:
                pbar.update(duracion_ms - cursor)                
            break
        # Buscar primer silencio tras el mínimo
        ventana = audio[after_min:duracion_ms]
        silencios = detect_silence(
            ventana,
            min_silence_len=min_silence_ms,
            silence_thresh=silence_thresh_dbfs
        )
        if silencios:
            # Usar el final del primer silencio y sumar cola configurable
            silence_start_rel, silence_end_rel = silencios[0]
            proposed_end = after_min + silence_end_rel + tail_after_silence_ms
            end = min(proposed_end, duracion_ms)
        else:
            end = duracion_ms
        # Ajuste de inicio: no aplicar tail_before_silence en el primer segmento
        modifiedStart = start
        if start > 0 and modifiedStart - tail_before_silence_ms >= 0:
            modifiedStart -= tail_before_silence_ms
        else:
            modifiedStart = max(0, modifiedStart)
        segmentos.append(audio[modifiedStart:end])
        if not maximo_cortes:
            pbar.update(end - cursor)
        else:
            pbar.update(len(segmentos))
        cursor = end
    pbar.close()
    print(f"Segmentación completada en {len(segmentos)} segmentos.")

    # Segunda pasada: recortar bordes y exportar cada segmento
    contador = 1
    for idx, seg in enumerate(tqdm(segmentos, desc="Recortando y exportando", unit="segmento", leave=False)):
        # En el primer segmento solo recortar el silencio final para no cortar el inicio real
        if idx == 0:
            fragmento = _trim_trailing_silence_only(seg)
        else:
            fragmento = _trim_leading_trailing_silence(seg)
        if len(fragmento) >= min_segment_ms:
            out_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_{contador}.wav"
            out_path = os.path.join(SPLITTED_DIR, out_filename)
            fragmento.export(out_path, format="wav")
            contador += 1

    print("Segmentos exportados en audio/splitted.")

def transcribir_audios(language: str = "es-ES"):
    # Recorre audio/splitted, transcribe y escribe metadata.csv
    os.makedirs(SPLITTED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH) or ".", exist_ok=True)
    recognizer = sr.Recognizer()

    archivos = [
        f for f in os.listdir(SPLITTED_DIR)
        if f.lower().endswith((".wav", ".flac", ".ogg"))
    ]
    if not archivos:
        print("No hay archivos para transcribir en audio/splitted.")
        return

    with open(METADATA_PATH, "a", encoding="utf-8") as meta:
        for fname in tqdm(archivos, desc="Transcribiendo", unit="archivo", leave=False):
            full_path = os.path.join(SPLITTED_DIR, fname)
            nombre_base = os.path.splitext(fname)[0]
            try:
                with sr.AudioFile(full_path) as source:
                    audio_data = recognizer.record(source)
                    texto = recognizer.recognize_google(audio_data, language=language)
            except Exception:
                texto = ""
            meta.write(f"{nombre_base}|{texto.strip()}\n")
    print("Transcripción completada y metadata.csv actualizado.")

def segmentar_todos_audios_por_palabras(
    dir: str = "audio/clean",
    min_segment_ms: int = 8000,
    min_silence_ms: int = 100,
    treshold_silencio_level: int = SILENCE_THRESH_DBFS,
    maximo_cortes=None,
    tail_after_silence_ms: int = 200,
    tail_before_silence_ms: int = 200
):
    """
    Recorre todos los audios en audio/clean y segmenta cada uno con segmentar_audio_por_palabras.
    """
    if not os.path.isdir(dir):
        print(f"No existe el directorio: {dir}")
        return

    archivos = [
        f for f in os.listdir(dir)
        if f.lower().endswith((".wav", ".flac", ".ogg", ".mp3", ".m4a"))
    ]
    if not archivos:
        print("No hay archivos de audio en audio/clean.")
        return

    for fname in tqdm(archivos, desc="Segmentando audios", unit="archivo", leave=False):
        path = os.path.join(dir, fname)
        try:
            segmentar_audio_por_palabras(
                path,
                min_segment_ms=min_segment_ms,
                min_silence_ms=min_silence_ms,
                silence_thresh_dbfs=treshold_silencio_level,
                maximo_cortes=maximo_cortes,
                tail_after_silence_ms=tail_after_silence_ms,
                tail_before_silence_ms=tail_before_silence_ms
            )
        except Exception as e:
            print(f"Error segmentando {fname}: {e}")
# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) != 3:
#         print("Uso: python segmentar_audio.py <archivo_wav> <num_palabras>")
#         exit(1)
#     audio_path = sys.argv[1]
#     num_palabras = int(sys.argv[2])
#     segmentar_audio_por_palabras(audio_path, num_palabras)
