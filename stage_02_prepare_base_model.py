import os
import urllib.request as request
import zipfile
import tensorflow as tf
from pathlib import Path
from dataclasses import dataclass
import yaml

# --- 1. CONFIGURATION ENTITY ---
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int

# --- 2. CONFIGURATION MANAGER ---
class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml", params_filepath="params.yaml"):
        # Load config and params
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(params_filepath, 'r') as f:
            self.params = yaml.safe_load(f)

        # Create directory for artifacts
        os.makedirs(self.config['artifacts_root'], exist_ok=True)

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config['prepare_base_model']
        
        # Create directory for this stage
        os.makedirs(config['root_dir'], exist_ok=True)

        return PrepareBaseModelConfig(
            root_dir=Path(config['root_dir']),
            base_model_path=Path(config['base_model_path']),
            updated_base_model_path=Path(config['updated_base_model_path']),
            params_image_size=self.params['IMAGE_SIZE'],
            params_learning_rate=self.params['LEARNING_RATE'],
            params_include_top=self.params['INCLUDE_TOP'],
            params_weights=self.params['WEIGHTS'],
            params_classes=self.params['CLASSES']
        )

# --- 3. THE COMPONENT (The Worker) ---
class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """
        Downloads VGG16 from Keras (TensorFlow)
        """
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )
        
        # Save the raw VGG16 model to artifacts
        self.save_model(path=self.config.base_model_path, model=self.model)
        print(f"Base VGG16 model saved to {self.config.base_model_path}")

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        """
        Modifies the model to work for our custom Poultry Data
        """
        # Freeze the pre-trained layers so we don't lose the "knowledge"
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        # Add our custom "Tail" (Output layers)
        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        # Combine them into a new model
        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        # Compile the model
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )
        
        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        print(f"Updated model saved to {self.config.updated_base_model_path}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

# --- 4. EXECUTION ---
if __name__ == '__main__':
    try:
        config_manager = ConfigurationManager()
        prepare_base_model_config = config_manager.get_prepare_base_model_config()
        
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        
        # Load VGG16
        prepare_base_model.get_base_model()
        # Modify it for Poultry
        prepare_base_model.update_base_model()
        
        print("\n>>> Stage 2: Prepare Base Model completed! <<<")
        
    except Exception as e:
        print(e)