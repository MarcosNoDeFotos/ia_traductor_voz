import os
import torch
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer

MODEL_PATH = "models/generated/"
OUTPUT_PATH = "output/generated_audio.wav"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# Cargar configuración original del entrenamiento
from_json_path = os.path.join(MODEL_PATH, "config.json")
config = GlowTTSConfig()
config.load_json(from_json_path)

# Inicializar AudioProcessor
ap = AudioProcessor.init_from_config(config)

# Inicializar tokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

# Inicializar modelo
model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
checkpoint = torch.load(MODEL_PATH + "best_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model"])
model.eval()

def generar_audio(texto, nombre_archivo, speaker_name=None, language=None):
    ids = tokenizer.encode(texto)
    input_ids = torch.LongTensor([ids])  # shape: [1, seq_len]
    input_lengths = torch.LongTensor([len(ids)])  # shape: [1]
    with torch.no_grad():
        mel_out = model.inference(x=input_ids, aux_input={"x_lengths": input_lengths})
        mel_tensor = mel_out["model_outputs"] if isinstance(mel_out, dict) else mel_out
        mel_numpy = mel_tensor.squeeze(0).cpu().numpy().T  # Transponer para [n_mel_channels, T]
        wav = ap.inv_melspectrogram(mel_numpy)
    ap.save_wav(wav, nombre_archivo)
    print(f"Audio generado: {nombre_archivo}")

if __name__ == "__main__":
    generar_audio(
        "prueba de texto a voz utilizando el modelo entrenado.",
        OUTPUT_PATH
    )