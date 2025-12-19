import os
import torch
from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

from utils.shared_configs import audio_config
def train_vocoder():
    output_path = os.path.join(os.path.dirname(__file__), "output/vocoder")
    data_path = os.path.join(os.path.dirname(__file__), "data")

    # checkpoint_path = None
    # # Buscar el último directorio de run en output
    # runs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("run-")]
    # if runs:
    #     runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_path, d)), reverse=True)
    #     last_run_dir = os.path.join(output_path, runs[0])
    #     # Buscar el último checkpoint en ese directorio (puede estar en subcarpeta "checkpoints" o en el root)

    #     checkpoints = [os.path.join(last_run_dir, f) for f in os.listdir(last_run_dir) if f.startswith("checkpoint") and f.endswith(".pth")]
    #     if checkpoints:
    #         checkpoints.sort(key=os.path.getmtime, reverse=True)
    #         checkpoint_path = checkpoints[0]


    config = HifiganConfig(
        audio=audio_config,
        batch_size=8,
        eval_batch_size=4,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=5,
        epochs=500,
        seq_len=8192,
        pad_short=2000,
        use_noise_augment=False,
        eval_split_size=4,
        print_step=5,
        print_eval=False,
        mixed_precision=True,
        lr_gen=1e-4,
        lr_disc=1e-4,
        data_path=os.path.join(data_path, "wavs"),
        output_path=output_path,
        save_checkpoints=True,
        save_step=25
    )
    ap = AudioProcessor(**config.audio.to_dict())
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)
    model = GAN(config, ap).to("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(
        TrainerArgs(gpu=0), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples, gpu=0
    )

    # if checkpoint_path: 
    #     optimizer = trainer.optimizer  # El optimizador se crea en Trainer
    #     print(f"Restaurando desde checkpoint: {checkpoint_path}")
    #     try:
    #         scaler = getattr(trainer, "scaler", None)
    #         trainer.restore_model(config, checkpoint_path, model, optimizer, scaler)
    #         import torch as th
    #         checkpoint_data = th.load(checkpoint_path, map_location="cpu")
    #         if "epoch" in checkpoint_data:
    #             # Mostrar advertencia si no se puede asignar el epoch
    #             print(f"Epoch encontrado en checkpoint ({checkpoint_data['epoch']}), pero Trainer no tiene atributo epoch conocido. El entrenamiento continuará desde el epoch 0.")
    #     except Exception as e:
    #         print(f"Error restaurando el checkpoint completo: {e}")
    #         print("Intentando restaurar solo los pesos del modelo (no se restaurará el optimizador ni el scaler).")
    #         model.load_checkpoint(config, checkpoint_path)
    # else:
    #     print("No se encontró ningún checkpoint en el último run. El entrenamiento empezará desde cero.")

    trainer.fit()

if __name__ == "__main__":
    train_vocoder()