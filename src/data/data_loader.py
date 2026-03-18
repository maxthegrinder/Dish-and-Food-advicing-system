import os
import pandas as pd
import kagglehub

class KaggleDataLoader:
    """
    Class responsible for downloading and loading the dataset from Kaggle.
    Demonstrates the Single Responsibility Principle (SRP).
    """
    def __init__(self, dataset_identifier: str):
        self.dataset_identifier = dataset_identifier
        self.dataset_path = None

    def download(self) -> str:
        """Downloads the dataset using kagglehub and returns the local path."""
        print(f"Downloading dataset: {self.dataset_identifier}...")
        self.dataset_path = kagglehub.dataset_download(self.dataset_identifier)
        print(f"Dataset downloaded to: {self.dataset_path}")
        return self.dataset_path

    def load_csv(self, file_name: str) -> pd.DataFrame:
        """Loads a CSV file from the downloaded dataset directory."""
        if not self.dataset_path:
            raise ValueError("Dataset path is not set. Call download() first.")
        file_path = os.path.join(self.dataset_path, file_name)
        print(f"Loading data from {file_path}...")
        return pd.read_csv(file_path)

    def get_images_dir(self, sub_dir: str = 'Food Images/Food Images') -> str:
        """Constructs the path to the images directory."""
        if not self.dataset_path:
            raise ValueError("Dataset path is not set. Call download() first.")
        return os.path.join(self.dataset_path, sub_dir)
