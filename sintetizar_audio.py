from TTS.api import TTS
import os
from datetime import datetime

MODELS_PATH = "models/generated/"  # Ruta a tu modelo entrenado
MODEL_PATH = MODELS_PATH +"glowtts/best_model.pth"  # Ruta a tu modelo entrenado
CONFIG_PATH = MODELS_PATH +"glowtts/config.json"     # Config de tu modelo
VOCODER_PATH = MODELS_PATH + "vocoder/best_model.pth"  # Ruta a tu vocoder entrenado
VOCODER_CONFIG_PATH = MODELS_PATH + "vocoder/config.json"  # Ruta a tu vocoder entrenado
# VOCODER_PATH = "C:\\Users\\MarcosNoDeFotos\\AppData\\Local\\tts\\vocoder_models--universal--libri-tts--wavegrad/model_file.pth"  # Ruta a tu vocoder entrenado
# VOCODER_CONFIG_PATH = "C:\\Users\\MarcosNoDeFotos\\AppData\\Local\\tts\\vocoder_models--universal--libri-tts--wavegrad/config.json"  # Ruta a tu vocoder entrenado


OUTPUT_PATH = "output"




def sintetizar_audio(texto: str, output_path: str = OUTPUT_PATH, modelo_path: str = MODEL_PATH, config_path: str = CONFIG_PATH, vocoder_path: str = VOCODER_PATH, vocoder_config_path: str = VOCODER_CONFIG_PATH):
    # Añadir fecha y hora al nombre del archivo de salida
    fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = os.path.join(output_path, f"sintetizado_{fecha_actual}.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Cargar modelo con TTS.api, indicando config y checkpoint
    tts = TTS(model_path=modelo_path, config_path=config_path, vocoder_path=vocoder_path, vocoder_config_path=vocoder_config_path)

    # Generar audio
    tts.tts_to_file(text=texto, file_path=output_path)

    print(f"Audio generado: {output_path}")
    return output_path