import os
from datetime import datetime
import json
import torch
import librosa
import soundfile as sf

from TTS.config import load_config
from TTS.utils.audio import AudioProcessor
from TTS.utils.synthesizer import Synthesizer
from utils.functions import copy_latest_vocoder_models

def test_vocoder(
    wav_path: str,
    vocoder_path: str = "models/generated/vocoder/best_model.pth",
    vocoder_config_path: str = "models/generated/vocoder/config.json",
    output_dir: str = "output/vocoder_test",
):
    copy_latest_vocoder_models()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1️⃣ Cargar config y AudioProcessor DEL VOCODER
    config = load_config(vocoder_config_path)

    checkpoint_data = torch.load(vocoder_path, map_location="cpu")
    step = "unknown"
    if "step" in checkpoint_data:
        # Mostrar advertencia si no se puede asignar el step
        step = f"{checkpoint_data['step']}"

    ap = AudioProcessor.init_from_config(config)

    # 2️⃣ Cargar WAV con el sample rate correcto
    wav, _ = librosa.load(wav_path, sr=ap.sample_rate)

    # 3️⃣ Generar MEL (FORMA CORRECTA EN TTS 0.22.x)
    mel = ap.melspectrogram(wav)          # [n_mels, T]
    mel = torch.FloatTensor(mel).unsqueeze(0).to(device)  # [1, n_mels, T]

    print("Mel shape:", mel.shape)
    print("Mel range:", mel.min().item(), mel.max().item())

    # 4️⃣ Cargar SOLO el vocoder
    synth = Synthesizer(
        tts_checkpoint=None,
        tts_config_path=None,
        vocoder_checkpoint=vocoder_path,
        vocoder_config=vocoder_config_path,
        use_cuda=(device == "cuda"),
    )



    vocoder = synth.vocoder_model.to(device).eval()

    # 5️⃣ Inferencia
    with torch.no_grad():
        wav_out = vocoder.inference(mel)

    wav_out = wav_out.squeeze().cpu().numpy()

    # 6️⃣ Guardar resultado
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.splitext(os.path.basename(wav_path))[0]
    out_path = os.path.join(output_dir, f"vocoder_{ts}_e{step}.wav")

    sf.write(out_path, wav_out, ap.sample_rate)
    print(f"Audio generado por vocoder: {out_path}")

    return out_path


if __name__ == "__main__":
    test_vocoder("data/wavs/frases_v1-001.wav")
