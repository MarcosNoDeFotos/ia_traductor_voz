import os
from scripts.extraer_audio_de_mp4 import extraer_pistas_audio
from scripts.clean_audio import clean_all_files
from scripts.segmentar_audio import segmentar_audio_por_palabras
from scripts.resamplear_audios_clean import resample_all_cleaned_files
from scripts.voice_trainer import train_voice_model
CURRENT_PATH = os.path.dirname(__file__).replace('\\', '/') + '/'
if __name__ == "__main__":
    # print(CURRENT_PATH)
    # extraer_pistas_audio(CURRENT_PATH + "videos/2025-12-09 18-31-53.mp4", extract_only=2) 
    # clean_all_files()


    # segmentar_audio_por_palabras(CURRENT_PATH + "audio/clean/2025-12-09 18-31-53_2.wav", num_palabras=12)


    # resample_all_cleaned_files()

    train_voice_model()