from TTS.utils.manage import ModelManager
from TTS.config import load_config
from TTS.tts.models.glow_tts import GlowTTS
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
import torch
import numpy as np
import cv2

TEXT = "this is a test of a model pretrained mel spectrogram"
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1️⃣ Descargar / localizar modelo
manager = ModelManager()
model_path, config_path, _ = manager.download_model(
    "tts_models/en/ljspeech/glow-tts"
)

# 2️⃣ Cargar config REAL del modelo
config = load_config(config_path)

# 3️⃣ Audio + tokenizer EXACTOS del modelo
ap = AudioProcessor.init_from_config(config)
tokenizer, config = TTSTokenizer.init_from_config(config)

# 4️⃣ Crear modelo
model = GlowTTS(config, ap, tokenizer)
model.load_checkpoint(config, model_path)
model.to(device)
model.eval()

# 5️⃣ Inferencia
tokens = tokenizer.text_to_ids(TEXT)
tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
lengths = torch.LongTensor([tokens.shape[1]]).to(device)

with torch.no_grad():
    outputs = model.inference(tokens, aux_input={"x_lengths": lengths})
    mel = outputs["model_outputs"][0].cpu().numpy()

# Normalizar el mel a rango [0,255] para visualizar como imagen de líneas
mel_norm = mel - mel.min()
if mel_norm.max() > 0:
    mel_norm = mel_norm / mel_norm.max()
img_height = 200  # altura de la imagen
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
signal_img_bgr = cv2.resize(signal_img_bgr, (img_width * scale_factor*10, img_height * scale_factor), interpolation=cv2.INTER_NEAREST)

cv2.imshow("Mel de modelo preentrenado GlowTTS", signal_img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("output/mel_modelo_preentrenado.png", signal_img_bgr)
