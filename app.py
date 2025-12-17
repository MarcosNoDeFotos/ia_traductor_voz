import os
from utils.extraer_audio_de_mp4 import extraer_pistas_audio, extraer_pistas_audio_full_dir
from utils.clean_audio import clean_all_files
from utils.segmentar_audio import segmentar_audio_por_palabras, clean_files
from utils.resamplear_audios_clean import resample_all_cleaned_files
from train_model import train_voice_model
from train_vocoder import train_vocoder
# from scripts.voice_trainer import train_voice_model
CURRENT_PATH = os.path.dirname(__file__).replace('\\', '/') + '/'

NUMERO_PALABRAS_SEGMENTAR = 6
DURACION_MINIMA_SEGMENTO_S = 3
DURACION_MAXIMA_SEGMENTO_S = 8
if __name__ == "__main__":
    # print(CURRENT_PATH)
    # extraer_pistas_audio(CURRENT_PATH + "videos/2025-12-09 18-31-53.mp4", extract_only=2, output_format="wav") 
    # extraer_pistas_audio_full_dir(extract_only=2, output_format="wav")
    # clean_all_files(1200)

    # segmentar_todos_audios_por_palabras(maximo_cortes=1, tail_after_silence_ms=500, tail_before_silence_ms=900, min_segment_ms=6000)
    # transcribir_audios()
    # for f in os.listdir(CURRENT_PATH + "audio/raw/"):
    #     if f.endswith(".wav"):
    #         print(f"Segmentando {f}...")
    #         segmentar_audio_por_palabras(NUMERO_PALABRAS_SEGMENTAR, [CURRENT_PATH + "audio/raw/" + f], DURACION_MINIMA_SEGMENTO_S, DURACION_MAXIMA_SEGMENTO_S)
    # segmentar_audio_por_palabras(CURRENT_PATH + "audio/clean/2025-12-09 18-31-53_2.wav")

    # resample_all_cleaned_files()


    train_voice_model()
    # train_vocoder()