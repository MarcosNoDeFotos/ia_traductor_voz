import os
from multiprocessing import freeze_support

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def train_voice_model():
    output_path = os.path.join(os.path.dirname(__file__), "../output")
    data_path = os.path.join(os.path.dirname(__file__), "../data/")
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", meta_file_train="metadata.csv", path=data_path, language="es"
    )
    config = GlowTTSConfig(
        batch_size=2,                # Reducir tamaño de lote
        eval_batch_size=4,           # Reducir tamaño de lote de evaluación
        num_loader_workers=2,        # Menos workers
        num_eval_loader_workers=2,   # Menos workers
        run_eval=False,              # Desactivar evaluación para ahorrar tiempo
        test_delay_epochs=-1,
        epochs=5,                    # Solo 5 épocas para pruebas rápidas
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="es-es",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=10,               # Imprimir más seguido para ver progreso
        print_eval=False,
        eval_split_size=0.017241379310344827,         # Usar la mitad de los datos para eval (reduce datos de entrenamiento)
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

    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    trainer = Trainer(
        
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    trainer.fit()