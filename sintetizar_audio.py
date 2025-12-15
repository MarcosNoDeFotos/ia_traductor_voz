from TTS.api import TTS
import os
from datetime import datetime

MODELS_PATH = "models/generated/"  # Ruta a tu modelo entrenado
MODEL_PATH = MODELS_PATH +"glowtts/best_model.pth"  # Ruta a tu modelo entrenado
CONFIG_PATH = MODELS_PATH +"glowtts/config.json"     # Config de tu modelo
VOCODER_PATH = MODELS_PATH + "vocoder/best_model.pth"  # Ruta a tu vocoder entrenado
VOCODER_CONFIG_PATH = MODELS_PATH + "vocoder/config.json"  # Ruta a tu vocoder entrenado

# Añadir fecha y hora al nombre del archivo de salida
fecha_actual = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_PATH = f"output/generated_audio_custom_{fecha_actual}.wav"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Cargar modelo con TTS.api, indicando config y checkpoint
tts = TTS(model_path=MODEL_PATH, config_path=CONFIG_PATH, vocoder_path=VOCODER_PATH, vocoder_config_path=VOCODER_CONFIG_PATH)

# Generar audio
texto = "Hola, esto es una prueba de mi modelo entrenado. \ndespués de un rato, se dirán más cosas interesantes."
tts.tts_to_file(text=texto, file_path=OUTPUT_PATH)

print(f"Audio generado: {OUTPUT_PATH}")
