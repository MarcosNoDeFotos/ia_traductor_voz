import os
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent
from tqdm import tqdm
from time import sleep
import matplotlib.pyplot as plt
import re
import shutil
from datetime import datetime

METADATA_PATH = "data/metadata.csv"
SPLITTED_DIR = "audio/splitted"

# Parámetros de silencio (ajustables)
MIN_SILENCE_LEN_MS = 300
SILENCE_THRESH_DBFS = -30


def _trim_leading_trailing_silence(seg: AudioSegment) -> AudioSegment:
    # Detecta regiones no silenciosas y recorta a ese rango
    ranges = detect_nonsilent(
        seg, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESH_DBFS
    )
    if not ranges:
        return AudioSegment.silent(duration=0, frame_rate=seg.frame_rate)
    start, end = ranges[0][0], ranges[-1][1]
    return seg[start:end]


def _trim_trailing_silence_only(seg: AudioSegment) -> AudioSegment:
    # Recorta solo el silencio al final, mantiene el inicio intacto
    ranges = detect_nonsilent(
        seg, min_silence_len=MIN_SILENCE_LEN_MS, silence_thresh=SILENCE_THRESH_DBFS
    )
    if not ranges:
        return AudioSegment.silent(duration=0, frame_rate=seg.frame_rate)
    end = ranges[-1][1]
    return seg[:end]


# Helper: convierte AudioSegment en sr.AudioData (mono, 16-bit, 16 kHz por defecto)
def audiosegment_to_audio_data(
    seg: AudioSegment, target_rate: int = 16000
) -> sr.AudioData:
    mono16 = seg.set_channels(1).set_sample_width(2).set_frame_rate(target_rate)
    return sr.AudioData(mono16.raw_data, mono16.frame_rate, mono16.sample_width)


# Nuevo helper: reconocimiento con reintentos y backoff ante throttling/errores
def safe_recognize_google(
    recognizer: sr.Recognizer,
    audio_data: sr.AudioData,
    language: str = "es-ES",
    max_retries: int = 3,
    backoff_base: float = 1.5,
) -> str:
    for attempt in range(max_retries):
        try:
            return recognizer.recognize_google(audio_data, language=language)
        except sr.UnknownValueError:
            # Audio no entendible
            return ""
        except sr.RequestError as e:
            # Throttling/red de Google; reintentar con backoff
            if attempt < max_retries - 1:
                sleep(backoff_base ** attempt)
                continue
            print(f"Error del servicio de Google Speech: {e}")
            return ""


def transcribir_audios(language: str = "es-ES"):
    # Recorre audio/splitted, transcribe y escribe metadata.csv
    os.makedirs(SPLITTED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH) or ".", exist_ok=True)
    recognizer = sr.Recognizer()

    archivos = [
        f
        for f in os.listdir(SPLITTED_DIR)
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


def normalizarSegmento(
    original: AudioSegment, start: int, end: int
) -> tuple[AudioSegment, int, int]:
    # Reemplaza los silencios en cualquier posición por silencios predeterminados de duración fija
    SILENCE_TRESHOLD = 30  # umbral de amplitud de muestra para silencio (±)
    DURACION_SILENCIO = 50  # duración del silencio requerida en ms
    PASO_MS = 100  # pasos de búsqueda (ms)

    duracion_total = len(original)

    def es_silencio_ms(seg: AudioSegment) -> bool:
        # True si todas las muestras están dentro del umbral
        samples = seg.get_array_of_samples()
        return all(-SILENCE_TRESHOLD <= v <= SILENCE_TRESHOLD for v in samples)

    # Ajustar inicio: buscar silencio hacia atrás
    new_start = max(0, start)
    while new_start > 0:
        win_start = max(0, new_start - DURACION_SILENCIO)
        win_end = new_start
        ventana = original[win_start:win_end]
        if len(ventana) < DURACION_SILENCIO:
            break
        if es_silencio_ms(ventana):
            # anclar al inicio del silencio encontrado
            new_start = win_start
            break
        # retroceder en pasos de 100ms
        new_start = max(0, new_start - PASO_MS)

    # Ajustar fin: buscar silencio hacia delante
    new_end = min(duracion_total, end)
    while new_end < duracion_total:
        win_start = new_end
        win_end = min(duracion_total, new_end + DURACION_SILENCIO)
        ventana = original[win_start:win_end]
        if len(ventana) < DURACION_SILENCIO:
            break
        if es_silencio_ms(ventana):
            # anclar al final del silencio encontrado
            new_end = win_end
            break
        # avanzar en pasos de 100ms
        new_end = min(duracion_total, new_end + PASO_MS)

    # Asegurar consistencia
    if new_end < new_start:
        new_start, new_end = start, end

    return new_start, new_end


def strip_silencio(segmento: AudioSegment):
    # Elimina silencios al inicio y al final del segmento
    SILENCE_TRESHOLD = 30
    DURACION_DETECCION_SILENCIO = 150  # ms mínimos de silencio para recortar
    DURACION_SILENCIO = 100  # ms de silencio a conservar en cada extremo
    MS_STEP = 1  # resolución de escaneo en ms

    dur_ms = len(segmento)
    if dur_ms == 0:
        return segmento

    def es_silencio(ms_ini: int, ms_fin: int) -> bool:
        sub = segmento[ms_ini:ms_fin]
        if len(sub) == 0:
            return False
        return all(
            -SILENCE_TRESHOLD <= v <= SILENCE_TRESHOLD
            for v in sub.get_array_of_samples()
        )

    # Medir silencio al inicio
    lead = 0
    while lead + MS_STEP <= dur_ms and es_silencio(lead, lead + MS_STEP):
        lead += MS_STEP

    # Medir silencio al final
    trail = 0
    while trail + MS_STEP <= dur_ms and es_silencio(
        dur_ms - trail - MS_STEP, dur_ms - trail
    ):
        trail += MS_STEP

    # Calcular recortes dejando solo DURACION_SILENCIO si se supera el umbral de detección
    new_start = 0
    if lead >= DURACION_DETECCION_SILENCIO:
        keep_start = min(DURACION_SILENCIO, lead)
        new_start = lead - keep_start

    new_end = dur_ms
    if trail >= DURACION_DETECCION_SILENCIO:
        keep_end = min(DURACION_SILENCIO, trail)
        new_end = dur_ms - (trail - keep_end)

    # Asegurar que no se invierten los límites
    new_start = max(0, min(new_start, dur_ms))
    new_end = max(0, min(new_end, dur_ms))
    if new_end <= new_start:
        return AudioSegment.silent(duration=0, frame_rate=segmento.frame_rate)

    return segmento[new_start:new_end]


def cleanText(text: str) -> str:
    # Normalizar tildes y ñ
    reemplazos = str.maketrans(
        {
            "á": "a",
            "é": "e",
            "í": "i",
            "ó": "o",
            "ú": "u",
            "Á": "A",
            "É": "E",
            "Í": "I",
            "Ó": "O",
            "Ú": "U",
            "ü": "u",
            "Ü": "U",
            "ç": "c",
            "Ç": "C",
            "+": "mas",
            "-": "menos",
            "*": "por",
            "%": " por cien",
            "$": "dolares",
            "€": "euros",
        }
    )
    t = text.translate(reemplazos)
    t = t.replace("ñ", "ni").replace("Ñ", "Ni")

    # Conversión de números (0-9999) a palabras en español
    unidades = [
        "cero",
        "uno",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
    ]
    especiales = {
        10: "diez",
        11: "once",
        12: "doce",
        13: "trece",
        14: "catorce",
        15: "quince",
        16: "dieciseis",
        17: "diecisiete",
        18: "dieciocho",
        19: "diecinueve",
        20: "veinte",
        21: "veintiuno",
        22: "veintidos",
        23: "veintitres",
        24: "veinticuatro",
        25: "veinticinco",
        26: "veintiseis",
        27: "veintisiete",
        28: "veintiocho",
        29: "veintinueve",
    }
    decenas = [
        "",
        "diez",
        "veinte",
        "treinta",
        "cuarenta",
        "cincuenta",
        "sesenta",
        "setenta",
        "ochenta",
        "noventa",
    ]
    centenas = [
        "",
        "cien",
        "doscientos",
        "trescientos",
        "cuatrocientos",
        "quinientos",
        "seiscientos",
        "setecientos",
        "ochocientos",
        "novecientos",
    ]

    def num_a_palabras(n: int) -> str:
        if n < 10:
            return unidades[n]
        if 10 <= n < 30:
            return especiales[n]
        if 30 <= n < 100:
            d, u = divmod(n, 10)
            return decenas[d] if u == 0 else f"{decenas[d]} y {unidades[u]}"
        if 100 <= n < 200:
            r = n - 100
            return "cien" if r == 0 else f"ciento {num_a_palabras(r)}"
        if 200 <= n < 1000:
            c, r = divmod(n, 100)
            return centenas[c] if r == 0 else f"{centenas[c]} {num_a_palabras(r)}"
        if 1000 <= n < 2000:
            r = n - 1000
            return "mil" if r == 0 else f"mil {num_a_palabras(r)}"
        if 2000 <= n < 10000:
            m, r = divmod(n, 1000)
            mil = "mil" if m == 1 else f"{num_a_palabras(m)} mil"
            return mil if r == 0 else f"{mil} {num_a_palabras(r)}"
        return str(n)  # fallback for mayores a 9999

    def reemplazar_num(match: re.Match) -> str:
        try:
            return num_a_palabras(int(match.group(0)))
        except ValueError:
            return match.group(0)

    t = re.sub(r"\d+", reemplazar_num, t)

    return t


def reemplazar_silencios_largos(segmento: AudioSegment) -> AudioSegment:
    SILENCE_TRESHOLD = 30
    DURACION_DETECCION_SILENCIO = 150  # ms mínimos para considerar un silencio "largo"
    DURACION_SILENCIO = 1000  # ms a dejar como silencio comprimido
    MS_STEP = 1

    dur_ms = len(segmento)
    if dur_ms == 0:
        return segmento

    def es_silencio(ms_ini: int, ms_fin: int) -> bool:
        ms_ini = max(0, min(ms_ini, dur_ms))
        ms_fin = max(0, min(ms_fin, dur_ms))
        sub = segmento[ms_ini:ms_fin]
        if len(sub) == 0:
            return False
        return all(
            -SILENCE_TRESHOLD <= v <= SILENCE_TRESHOLD
            for v in sub.get_array_of_samples()
        )

    out = AudioSegment.silent(duration=0, frame_rate=segmento.frame_rate)
    out = out.set_sample_width(segmento.sample_width).set_channels(segmento.channels)

    i = 0
    while i < dur_ms:
        # Detectar bloque de silencio
        if es_silencio(i, min(dur_ms, i + MS_STEP)):
            run_start = i
            while i < dur_ms and es_silencio(i, min(dur_ms, i + MS_STEP)):
                i += MS_STEP
            run_end = i
            run_len = run_end - run_start

            if run_len >= DURACION_DETECCION_SILENCIO:
                # Sustituir por un silencio comprimido
                sl = AudioSegment.silent(
                    duration=DURACION_SILENCIO, frame_rate=segmento.frame_rate
                )
                sl = sl.set_sample_width(segmento.sample_width).set_channels(
                    segmento.channels
                )
                out += sl
            else:
                # Mantener el silencio corto original
                out += segmento[run_start:run_end]
        else:
            # Copiar bloque no silencioso
            non_start = i
            while i < dur_ms and not es_silencio(i, min(dur_ms, i + MS_STEP)):
                i += MS_STEP
            out += segmento[non_start:i]

    return out


def segmentar_audio_por_palabras(
    num_palabras: int,
    wav_paths: list[str],
    min_duration_sec: int = 1,
    max_duration_sec: int = 15,
):
    os.makedirs(SPLITTED_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(METADATA_PATH) or ".", exist_ok=True)

    recognizer = sr.Recognizer()
    corte_segmento = 4000

    # Backup de metadata.csv antes de entrar al for
    try:
        if os.path.isfile(METADATA_PATH):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = os.path.dirname(METADATA_PATH) or "."
            backup_path = os.path.join(backup_dir, f"metadata_{ts}.csv")
            shutil.copy2(METADATA_PATH, backup_path)
            print(f"Backup creado: {backup_path}")
        else:
            print("metadata.csv no existe; se omitió el backup inicial.")
    except Exception as e:
        print(f"No se pudo crear el backup de metadata.csv: {e}")

    for wav_path in wav_paths:
        audio = AudioSegment.from_wav(wav_path)
        len_original = len(audio)
        audio = strip_silencio(audio)
        print("Limpiando silencios largos...")
        audio = reemplazar_silencios_largos(audio)
        print(f"Duración original: {len_original / 1000}s, tras limpieza: {len(audio) / 1000}s.")
        duracion_ms = len(audio)
        position = 0
        last_end = 0
        contador = 0
        base_name = os.path.splitext(os.path.basename(wav_path))[0]
        while position < duracion_ms:
            if last_end != 0:
                end = min(last_end + corte_segmento / 2, duracion_ms)
            else:
                end = min(position + corte_segmento, duracion_ms)
            start, end = normalizarSegmento(audio, position, end)
            segmento: AudioSegment = audio[start:end]
            print(f"{end*100/duracion_ms:.2f}%. {start} - {end}.")
            # Usar sr.AudioData directamente sin exportar ni context manager
            audio_data = audiosegment_to_audio_data(segmento)

            # Reemplazo del reconocimiento directo por el helper con reintentos
            texto = safe_recognize_google(recognizer, audio_data, language="es-ES")
            # Pequeña pausa para evitar rate limiting agresivo
            sleep(0.2)

            texto = cleanText(texto)
            text_length = len(texto.split())
            if text_length >= num_palabras:
                segmento = reemplazar_silencios_largos(
                    segmento
                )  # Se eliminan los silencios que haya en cualquier posición por un silencio de 1s
                segmento = strip_silencio(
                    segmento
                )  # Se eliminan los silencios al inicio y al final del segmento
                segmento_length = len(segmento)
                if (
                    segmento_length >= min_duration_sec * 1000
                    and segmento_length <= max_duration_sec * 1000
                ):
                    out_filename = f"{base_name}_{contador}"
                    segmento.export(
                        os.path.join(SPLITTED_DIR, out_filename + ".wav"), format="wav"
                    )
                    with open(METADATA_PATH, "a", encoding="utf-8") as meta:
                        meta.write(f"{out_filename}|{texto}|{texto}\n")
                    print(
                        f"Exportado segmento {out_filename} con {text_length} palabras ({segmento_length/1000:.2f}s.) ({end}/{duracion_ms})"
                    )
                else:
                    print(
                        f"Segmento descartado por duración: {segmento_length/1000:.2f}s. ({end}/{duracion_ms})"
                    )
                contador += 1
                position = end
                last_end = 0
            elif text_length != 0: # Se reconoce algo pero no llega al mínimo
                last_end = end
            else:
                last_end = 0
                position = end
            if duracion_ms - last_end < corte_segmento:
                break
        print(f"Segmentación completada para {wav_path}, total segmentos: {contador}.")


def clean_files(min_duration_sec: int = 1, max_duration_sec: int = 15):
    # Elimina audios fuera de rango y sus entradas en metadata.csv
    if not os.path.isdir(SPLITTED_DIR):
        print(f"No existe el directorio: {SPLITTED_DIR}")
        return

    exts = (".wav", ".flac", ".ogg", ".mp3", ".m4a")
    archivos = [f for f in os.listdir(SPLITTED_DIR) if f.lower().endswith(exts)]
    if not archivos:
        print("No hay archivos en audio/splitted.")
        return

    eliminados = []
    for fname in archivos:
        fpath = os.path.join(SPLITTED_DIR, fname)
        try:
            seg = AudioSegment.from_file(fpath)
            dur_s = len(seg) / 1000.0
        except Exception:
            # Si no se puede leer, se elimina
            dur_s = 0.0

        if dur_s < min_duration_sec or dur_s > max_duration_sec:
            try:
                os.remove(fpath)
                eliminados.append(os.path.splitext(fname)[0])  # base sin extensión
                print(f"Eliminado {fname} (duración {dur_s:.2f}s).")
            except OSError as e:
                print(f"No se pudo eliminar {fname}: {e}")

    if not eliminados:
        print("No se eliminaron archivos por duración.")
        return

    # Actualizar metadata.csv eliminando líneas cuyo primer campo coincida
    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as fr:
                lineas = fr.readlines()
            with open(METADATA_PATH, "w", encoding="utf-8") as fw:
                for linea in lineas:
                    clave = linea.split("|", 1)[0].strip()
                    if clave not in eliminados:
                        fw.write(linea)
            print(
                f"Actualizado {METADATA_PATH}, eliminadas {len(eliminados)} entradas."
            )
        except OSError as e:
            print(f"Error actualizando {METADATA_PATH}: {e}")
    else:
        print("metadata.csv no existe; solo se eliminaron archivos.")
