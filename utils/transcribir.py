import os
import speech_recognition as sr

CLEAN_DIR = "data/clean"

def transcribir_audio(audio_path, txt_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        texto = recognizer.recognize_google(audio, language="es-ES")
    except Exception as e:
        texto = f"[ERROR]: {e}"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(texto)

if __name__ == "__main__":
    for file in os.listdir(CLEAN_DIR):
        if file.endswith(".wav"):
            audio_path = os.path.join(CLEAN_DIR, file)
            txt_path = os.path.join(CLEAN_DIR, f"{os.path.splitext(file)[0]}.txt")
            print(f"Transcribiendo {file}...")
            transcribir_audio(audio_path, txt_path)
    print("Transcripción completada.")
