import librosa
import numpy as np
import cv2

# y, sr = librosa.load('data/wavs/2025-12-09 18-31-53_2_1.wav')
y, sr = librosa.load('output/generated_audio_custom.wav')

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
scale_factor = 2
signal_img_bgr = cv2.resize(signal_img_bgr, (img_width * scale_factor, img_height * scale_factor), interpolation=cv2.INTER_NEAREST)

cv2.imshow("Signal", signal_img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("output/signal_generated_audio_custom.png", signal_img_bgr)