import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
import torch
import glob


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "output")
    data_path = os.path.join(os.path.dirname(__file__), "data")




    checkpoint_path = None
    # Buscar el último directorio de run en output
    runs = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("run-")]
    if runs:
        runs.sort(key=lambda d: os.path.getmtime(os.path.join(output_path, d)), reverse=True)
        last_run_dir = os.path.join(output_path, runs[0])
        # Buscar el último checkpoint en ese directorio (puede estar en subcarpeta "checkpoints" o en el root)

        checkpoints = [os.path.join(last_run_dir, f) for f in os.listdir(last_run_dir) if f.startswith("checkpoint") and f.endswith(".pth")]
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = checkpoints[0]




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
        eval_split_size=0.017241379310344827,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        save_checkpoints=True,
        datasets=[dataset_config],
        save_step=10,  # Guarda un checkpoint cada 100 pasos (ajusta según prefieras)
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
        TrainerArgs(gpu=0),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        gpu=0,
    )

    if checkpoint_path: 
        optimizer = trainer.optimizer  # El optimizador se crea en Trainer
        print(f"Restaurando desde checkpoint: {checkpoint_path}")
        try:
            scaler = getattr(trainer, "scaler", None)
            trainer.restore_model(config, checkpoint_path, model, optimizer, scaler)
            import torch as th
            checkpoint_data = th.load(checkpoint_path, map_location="cpu")
            if "epoch" in checkpoint_data:
                # Mostrar advertencia si no se puede asignar el epoch
                print(f"Epoch encontrado en checkpoint ({checkpoint_data['epoch']}), pero Trainer no tiene atributo epoch conocido. El entrenamiento continuará desde el epoch 0.")
        except Exception as e:
            print(f"Error restaurando el checkpoint completo: {e}")
            print("Intentando restaurar solo los pesos del modelo (no se restaurará el optimizador ni el scaler).")
            model.load_checkpoint(config, checkpoint_path)
    else:
        print("No se encontró ningún checkpoint en el último run. El entrenamiento empezará desde cero.")

    trainer.fit()