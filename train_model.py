import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import torch


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "output")
    data_path = os.path.join(os.path.dirname(__file__), "data3")
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=os.path.join(data_path)
    )
    config = GlowTTSConfig(
        batch_size=10,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en-us",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        # eval_split_size=0.017241379310344827,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
    )
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ADVERTENCIA: CUDA no está disponible. El entrenamiento se realizará en CPU.")
    else:
        print("CUDA disponible. El entrenamiento se realizará en GPU.")

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None).to(device)
    trainer = Trainer(
        TrainerArgs(gpu=0),  # Fuerza uso de CUDA en Trainer
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        gpu=0
    )
    trainer.fit()