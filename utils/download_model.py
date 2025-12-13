from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

model_name = "tts_models/multilingual/multi-dataset/your_tts"

manager = ModelManager()
model_path, config_path, model_item = manager.download_model(model_name)

print("Modelo descargado en:")
print(model_path)
print(config_path)
