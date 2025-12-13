import torchaudio

wav, sr = torchaudio.load("data/clean/2025-12-09 21-01-57_3_1.wav")
print(wav.shape, sr)