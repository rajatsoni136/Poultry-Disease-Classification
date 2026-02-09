import tensorflow as tf
from pathlib import Path
import yaml
from dataclasses import dataclass
import os
import json

# --- 1. CONFIGURATION ENTITY ---
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    params_image_size: list
    params_batch_size: int

# --- 2. CONFIGURATION MANAGER ---
class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml", params_filepath="params.yaml"):
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)
        with open(params_filepath, 'r') as f:
            self.params = yaml.safe_load(f)

    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = self.config['evaluation']
        
        return EvaluationConfig(
            path_of_model=Path(eval_config['path_of_model']),
            training_data=Path(eval_config['training_data']),
            params_image_size=self.params['IMAGE_SIZE'],
            params_batch_size=self.params['BATCH_SIZE']
        )

# --- 3. THE COMPONENT ---
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    def _valid_generator(self):
        datagenerator_kwargs = dict(
            rescale = 1./255,
            validation_split=0.20
        )
        
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        
        # Calculate scores
        self.score = self.model.evaluate(self.valid_generator)

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        
        # Save to scores.json
        with open("scores.json", "w") as f:
            json.dump(scores, f, indent=4)
        
        print(f"Json score saved: {scores}")

# --- 4. EXECUTION ---
if __name__ == '__main__':
    try:
        config = ConfigurationManager()
        val_config = config.get_evaluation_config()
        evaluation = Evaluation(val_config)
        
        evaluation.evaluation()
        evaluation.save_score()
        
        print(">>> Stage 4: Evaluation Completed! <<<")

    except Exception as e:
        print(e)