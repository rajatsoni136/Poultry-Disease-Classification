import os
import urllib.request as request
import zipfile
import yaml
from dataclasses import dataclass
from pathlib import Path

# --- 1. CONFIGURATION ENTITY ---
# This acts as a strictly typed variable holder so we don't make mistakes with file paths later.
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

# --- 2. CONFIGURATION MANAGER ---
# This class reads your config.yaml and prepares the directories.
class ConfigurationManager:
    def __init__(self, config_filepath="config.yaml"):
        # Load the YAML file
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)

        # Create the main 'artifacts' folder if it doesn't exist
        # (This corresponds to the bottom blue box in your diagram)
        artifacts_root = self.config['artifacts_root']
        os.makedirs(artifacts_root, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config['data_ingestion']

        # Create the specific 'data_ingestion' folder inside artifacts
        os.makedirs(config['root_dir'], exist_ok=True)

        # Return the clean configuration object
        return DataIngestionConfig(
            root_dir=Path(config['root_dir']),
            source_URL=config['source_URL'],
            local_data_file=Path(config['local_data_file']),
            unzip_dir=Path(config['unzip_dir'])
        )

# --- 3. THE COMPONENT (The Worker) ---
# This class actually does the work shown in Stage 1 of your diagram.
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        """
        Downloads the file from the URL specified in config.yaml
        """
        if not os.path.exists(self.config.local_data_file):
            print(f"Downloading data from {self.config.source_URL}...")
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            print(f"{filename} downloaded successfully!")
        else:
            print(f"File already exists at {self.config.local_data_file}")

    def extract_zip_file(self):
        """
        Unzips the downloaded file into the unzip_dir
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        print(f"Unzipping data to {unzip_path}...")
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        print("Unzipping completed!")

# --- 4. THE PIPELINE EXECUTION ---
if __name__ == '__main__':
    try:
        # Initialize Manager
        config_manager = ConfigurationManager()
        
        # Get Settings
        data_ingestion_config = config_manager.get_data_ingestion_config()
        
        # Initialize Data Ingestion Component
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # Run Steps
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        
        print("\n>>> Stage 1: Data Ingestion completed successfully! <<<")
        
    except Exception as e:
        print(e)