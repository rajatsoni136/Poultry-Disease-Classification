import os
import urllib.request as request
import zipfile
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
import yaml
# --- 1. CONFIGURATION ENTITY ---
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

# --- 2. CONFIGURATION MANAGER ---
class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml", params_filepath="params.yaml"):
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(params_filepath, 'r') as f:
            self.params = yaml.safe_load(f)

        os.makedirs(self.config['artifacts_root'], exist_ok=True)

    def get_training_config(self) -> TrainingConfig:
        training = self.config['training']
        prepare_base_model = self.config['prepare_base_model']
        params = self.params
        
        training_data = os.path.join(self.config['data_ingestion']['unzip_dir'], "Chicken-fecal-images")
        
        os.makedirs(training['root_dir'], exist_ok=True)

        return TrainingConfig(
            root_dir=Path(training['root_dir']),
            trained_model_path=Path(training['trained_model_path']),
            updated_base_model_path=Path(prepare_base_model['updated_base_model_path']),
            training_data=Path(training_data),
            params_epochs=params['EPOCHS'],
            params_batch_size=params['BATCH_SIZE'],
            params_is_augmentation=params['AUGMENTATION'],
            params_image_size=params['IMAGE_SIZE']
        )

# --- 3. THE COMPONENT (The Trainer) ---
class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )

    def train_valid_generator(self):
        # This setup splits your data: 80% for training, 20% for validation/testing
        datagenerator_kwargs = dict(
            rescale = 1./255, # Normalize pixel values
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Generator for Training Data
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        # Generator for Validation Data
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

    def train(self):
        # Calculate steps per epoch based on data size
        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # The actual training command
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator
        )

        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

# --- 4. EXECUTION ---
if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()
        
        training = Training(config=training_config)
        
        training.get_base_model()
        training.train_valid_generator()
        training.train()
        
        print("\n>>> Stage 3: Training completed! Model saved. <<<")
        
    except Exception as e:
        print(e)