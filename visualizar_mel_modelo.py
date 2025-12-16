import torch
import numpy as np
import cv2
import os
from datetime import datetime
import time

from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.config import load_config

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
    idGeneracion = 0
    while running:
        config = load_config(CONFIG_PATH)
        ap = AudioProcessor.init_from_config(config)
        tokenizer, config = TTSTokenizer.init_from_config(config)
        model = GlowTTS(config, ap, tokenizer)
        model.load_checkpoint(config, MODEL_PATH)
        model.to(device)
        model.eval()
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

            # Mostrar con cv2
            # cv2.imshow("Mel generado por modelo", signal_img_bgr)
            # cv2.waitKey(0)  # Mostrar durante 0.5 segundos (ajusta si quieres)

            output_dir = "output/mels_generados_modelo"
            os.makedirs(output_dir, exist_ok=True)
            fecha_actual = datetime.now().strftime("%Y-%m-%d_%H.%M") + f"_{idGeneracion}"
            idGeneracion += 1
            output_path = os.path.join(output_dir, f"{fecha_actual}.png")
            cv2.imwrite(output_path, signal_img_bgr)
            print(f"Mel spectrogram guardado como {output_path}")

            prev_mel = mel
        running = RUN_FOREVER
        if running:
            time.sleep(15)  # Esperar 15 segundos antes de la siguiente generación
    cv2.destroyAllWindows()
except KeyboardInterrupt:
    print("Finalizado por el usuario (Ctrl+C).")
    cv2.destroyAllWindows()
