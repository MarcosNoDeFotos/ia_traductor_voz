import os
import torch
from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN
if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "output")
    data_path = os.path.join(os.path.dirname(__file__), "data3")

    config = HifiganConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=5,
        epochs=10,
        seq_len=8192,
        pad_short=2000,
        use_noise_augment=True,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        lr_gen=1e-4,
        lr_disc=1e-4,
        data_path=os.path.join(data_path, "wavs"),
        output_path=output_path,
    )
    ap = AudioProcessor(**config.audio.to_dict())
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)
    model = GAN(config, ap).to("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )
    trainer.fit()