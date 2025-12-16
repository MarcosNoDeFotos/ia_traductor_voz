import torch
import numpy as np
import cv2
import os
from datetime import datetime
from sintetizar_audio import sintetizar_audio
import librosa
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.config import load_config
from utils.functions import copy_latest_run_models, copy_latest_vocoder_models
# RUTAS
MODEL_PATH = "./models/generated/glowtts/best_model.pth"
CONFIG_PATH = "./models/generated/glowtts/config.json"

# MODEL_PATH = "./output/run-December-13-2025_07+45PM-1d17f5a/best_model.pth"
# CONFIG_PATH = "./output/run-December-13-2025_07+45PM-1d17f5a/config.json"

TEXT = "Hola, esto es una prueba de mi modelo entrenado. \ndespués de un rato, se dirán más cosas interesantes."
RUN_FOREVER = False
device = "cuda" if torch.cuda.is_available() else "cpu"

# cargar config




prev_mel = None
running = True
try:
    copy_latest_run_models()
    copy_latest_vocoder_models()
    config = load_config(CONFIG_PATH)
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    model = GlowTTS(config, ap, tokenizer)
    model.load_checkpoint(config, MODEL_PATH)
    model.to(device)
    model.eval()

    checkpoint_data = torch.load(MODEL_PATH, map_location="cpu")
    step = "s_unknown"
    if "step" in checkpoint_data:
        # Mostrar advertencia si no se puede asignar el step
        step = f"s_{checkpoint_data['step']}"

    # texto → tokens
    tokens = tokenizer.text_to_ids(TEXT)
    tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
    tokens_lengths = torch.LongTensor([tokens.shape[1]]).to(device)

    # inferencia (mel)
    with torch.no_grad():
        outputs = model.inference(tokens, aux_input={"x_lengths": tokens_lengths})
        mel = outputs["model_outputs"][0].cpu().numpy()

    # Solo guarda si el mel es diferente al anterior
    if prev_mel is not None and np.array_equal(mel, prev_mel):
        # print("Mel idéntico al anterior, no se guarda.")
        None
    else:
        # Normalizar el mel a rango [0,255] para visualizar como imagen de líneas (como visualizar_wav_2)
        mel_norm = mel - mel.min()
        if mel_norm.max() > 0:
            mel_norm = mel_norm / mel_norm.max()
        img_height = 200
        img_width = mel.shape[1]
        signal_img = np.ones((img_height, img_width), dtype=np.uint8) * 255  # fondo blanco

        # Dibujar la señal de cada bin mel como una línea (por cada fila del mel)
        for mel_bin in range(mel.shape[0]):
            y_plot = (mel_norm[mel_bin] * (img_height - 1)).astype(np.int32)
            for x in range(img_width - 1):
                cv2.line(signal_img, (x, img_height - 1 - y_plot[x]), (x + 1, img_height - 1 - y_plot[x + 1]), 0, 1)

        # Convertir a BGR para mostrar con cv2
        signal_img_bgr = cv2.cvtColor(signal_img, cv2.COLOR_GRAY2BGR)

        # Aumentar tamaño si se desea
        scale_factor = 2
        signal_img_bgr = cv2.resize(signal_img_bgr, (img_width * scale_factor, img_height * scale_factor), interpolation=cv2.INTER_NEAREST)


        fecha_actual = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
        output_dir = f"output/mels_generados_modelo/{fecha_actual}_{step}"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"mel_modelo.png")
        cv2.imwrite(output_path, signal_img_bgr)
        print(f"Mel de modelo guardado como {output_path}")


        wav_path = sintetizar_audio(TEXT, output_path=output_dir, modelo_path=MODEL_PATH, config_path=CONFIG_PATH)

        y, sr = librosa.load(wav_path)

        # Normalizar la señal a rango [0,255] para visualizar
        y_norm = y - y.min()
        if y_norm.max() > 0:
            y_norm = y_norm / y_norm.max()
        img_height = 200  # altura de la imagen
        img_width = len(y_norm)
        signal_img = np.ones((img_height, img_width), dtype=np.uint8) * 255  # fondo blanco

        # Dibujar la señal en la imagen
        y_plot = (y_norm * (img_height - 1)).astype(np.int32)
        for x in range(img_width - 1):
            cv2.line(signal_img, (x, img_height - 1 - y_plot[x]), (x + 1, img_height - 1 - y_plot[x + 1]), 0, 1)

        # Convertir a BGR para mostrar con cv2
        signal_img_bgr = cv2.cvtColor(signal_img, cv2.COLOR_GRAY2BGR)

        # Aumentar tamaño si se desea
        signal_img_bgr = cv2.resize(signal_img_bgr, (img_width * scale_factor, img_height * scale_factor*2), interpolation=cv2.INTER_NEAREST)

        output_path = os.path.join(output_dir, f"mel_wav.png")
        cv2.imwrite(output_path, signal_img_bgr)
        print(f"Mel de wav sintetizado guardado como {output_path}")
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    print("Finalizado por el usuario (Ctrl+C).")
    cv2.destroyAllWindows()



# Sincronizar los artefactos del último entrenamiento al directorio 'models/generated/glowtts'
# copy_latest_run_models("output", "models/generated/glowtts")
