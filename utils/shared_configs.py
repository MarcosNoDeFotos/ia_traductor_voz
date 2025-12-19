from TTS.config.shared_configs import BaseAudioConfig

audio_config = BaseAudioConfig(
    sample_rate=22050,
    fft_size=1024,
    hop_length=256,
    win_length=1024,
)