import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt

wav, sr = librosa.load("data/wavs/2025-12-09 18-31-53_2_1.wav", sr=None)

mel = librosa.feature.melspectrogram(
    y=wav,
    sr=sr,
    n_mels=80,
    fmax=8000
)

mel_db = librosa.power_to_db(mel, ref=np.max)

# Normalizar el mel a rango [0,255] para visualizar con cv2
mel_norm = mel_db - mel_db.min()
if mel_norm.max() > 0:
    mel_norm = mel_norm / mel_norm.max()
mel_img = (mel_norm * 255).astype(np.uint8)

# Aplicar colormap de OpenCV
mel_img_color = cv2.applyColorMap(mel_img, cv2.COLORMAP_VIRIDIS)

# Aumentar tamaño si se desea (opcional)
scale_factor = 2
height, width = mel_img_color.shape[:2]
mel_img_color_resized = cv2.resize(mel_img_color, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST)

# Mostrar con cv2
cv2.imshow("Mel espectrograma del audio generado", mel_img_color_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Mostrar ejes y ticks para mejor interpretación
# plt.figure(figsize=(12, 5))
# plt.imshow(mel_db, aspect="auto", origin="lower", cmap="viridis")
# plt.title("Mel espectrograma (dB, generado)")
# plt.xlabel("Frames (tiempo)")
# plt.ylabel("Mel bins (frecuencia)")
# plt.colorbar(format="%+2.0f dB")
# plt.tight_layout()

