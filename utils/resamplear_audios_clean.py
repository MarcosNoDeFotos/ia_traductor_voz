# ffmpeg -i input.wav -ar 22050 -ac 1 output.wav
import os
import subprocess
def resample_audio(input_path, output_path, target_sr=22050):
    """
    Resample audio to target sample rate and convert to mono using ffmpeg.
    """
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', str(target_sr), '-ac', '1',
        output_path
    ]
    subprocess.run(cmd, check=True)
    print(f"Resampled {input_path} to {output_path} at {target_sr} Hz mono.")

def resample_all_cleaned_files():
    SRC_DIR = "audio/splitted"
    RESAMPLED_DIR = "data/wavs"
    os.makedirs(RESAMPLED_DIR, exist_ok=True)

    for file in os.listdir(SRC_DIR):
        if file.endswith(".m4a") or file.endswith(".mp3") or file.endswith(".wav"):
            base, ext = os.path.splitext(file)
            out_file = f"{base}.wav"
            resample_audio(
                os.path.join(SRC_DIR, file),
                os.path.join(RESAMPLED_DIR, out_file),
                target_sr=22050
            )
    print("Resampling completed.")