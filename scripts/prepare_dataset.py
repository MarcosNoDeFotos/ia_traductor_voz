import os
import csv

CLEAN_DIR = "data/clean"
META = "data/metadata.csv"

# Pide que cada audio tenga un archivo .txt con el texto leído
# Ejemplo: 001.wav → 001.txt
with open(META, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="|")

    for file in os.listdir(CLEAN_DIR):
        if file.endswith(".wav"):
            txt_path = os.path.join(CLEAN_DIR, file.replace(".wav", ".txt"))
            if os.path.exists(txt_path):
                text = open(txt_path, "r", encoding="utf-8").read().strip()
                writer.writerow([file, text, "yourvoice"])
            else:
                print("Falta texto para:", file)

print("Dataset preparado.")
