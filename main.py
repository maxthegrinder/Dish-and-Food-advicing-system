import subprocess
import sys
import os

print("Checking and installing dependencies from requirements.txt...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"])
    print("Dependencies installed successfully!\n")
except Exception as e:
    print(f"Failed to install dependencies: {e}")

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.data_loader import KaggleDataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.dataset_builder import DatasetBuilder
from src.models.text_models import TfIdfModel, EmbeddingModel
from src.models.vision_models import ResNet50Model, VGG16Model
from src.models.multimodal import MultimodalModel
from src.evaluation.evaluator import ModelEvaluator
from src.evaluation.visualizer import Visualizer

def run_pipeline():
    print("Starting Machine Learning Pipeline...")

    # 1. Data Loading (SRP)
    loader = KaggleDataLoader("pes12017000148/food-ingredients-and-recipe-dataset-with-images")
    loader.download()

    csv_file = 'Food Ingredients and Recipe Dataset with Image Name Mapping.csv'
    df = loader.load_csv(csv_file)
    images_dir = loader.get_images_dir()

    # 2. Data Preprocessing (SRP)
    preprocessor = DataPreprocessor(df, images_dir)
    preprocessor.generate_missing_cuisines()
    preprocessor.remove_missing_images()
    labels = preprocessor.encode_labels()

    num_classes = preprocessor.get_num_classes()
    class_names = preprocessor.get_class_names()

    X_tfidf, X_padded, tokenizer = preprocessor.preprocess_text_features()

    # Splitting Data (DIP: passing data into independent components)
    train_idx, test_idx = train_test_split(preprocessor.df.index, test_size=0.2, random_state=42, stratify=labels)

    train_df = preprocessor.df.loc[train_idx]
    test_df = preprocessor.df.loc[test_idx]
    y_train = labels.loc[train_idx].values
    y_test = labels.loc[test_idx].values

    # 3. Dataset Building (SRP)
    builder = DatasetBuilder(batch_size=32)

    train_resnet = builder.create_image_dataset(train_df, y_train, 'resnet')
    test_resnet = builder.create_image_dataset(test_df, y_test, 'resnet')

    train_vgg = builder.create_image_dataset(train_df, y_train, 'vgg')
    test_vgg = builder.create_image_dataset(test_df, y_test, 'vgg')

    train_multi = builder.create_multimodal_dataset(train_df, X_tfidf[train_idx], y_train)
    test_multi = builder.create_multimodal_dataset(test_df, X_tfidf[test_idx], y_test)

    # 4. Modeling (OCP/LSP) & Evaluation
    evaluator = ModelEvaluator(class_names)

    print("\n--- Training TF-IDF Model ---")
    tfidf_model = TfIdfModel(input_dim=X_tfidf.shape[1], num_classes=num_classes)
    history_tfidf = tfidf_model.train(X_tfidf[train_idx], y_train, X_tfidf[test_idx], y_test, epochs=10)
    evaluator.evaluate(tfidf_model, X_tfidf[test_idx], y_test, "TF-IDF MLP")

    print("\n--- Training Embeddings Model ---")
    emb_model = EmbeddingModel(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=100,
        input_length=X_padded.shape[1],
        num_classes=num_classes
    )
    history_emb = emb_model.train(X_padded[train_idx], y_train, X_padded[test_idx], y_test, epochs=10)
    evaluator.evaluate(emb_model, X_padded[test_idx], y_test, "Embedding MLP")

    print("\n--- Training ResNet50 Model ---")
    resnet_model = ResNet50Model(num_classes=num_classes)
    history_resnet = resnet_model.train(train_resnet, test_resnet, epochs=5)
    evaluator.evaluate(resnet_model, test_resnet, y_test, "ResNet50")

    print("\n--- Training VGG16 Model ---")
    vgg_model = VGG16Model(num_classes=num_classes)
    history_vgg = vgg_model.train(train_vgg, test_vgg, epochs=5)
    evaluator.evaluate(vgg_model, test_vgg, y_test, "VGG16")

    print("\n--- Training Multimodal Model ---")
    multi_model = MultimodalModel(
        text_input_dim=X_tfidf.shape[1],
        image_input_shape=(224, 224, 3),
        num_classes=num_classes
    )
    history_multi = multi_model.train(train_multi, test_multi, epochs=5)
    evaluator.evaluate(multi_model, test_multi, y_test, "Multimodal Model")

    print("\nPipeline finished successfully!")

if __name__ == "__main__":
    # Disable GPU warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    run_pipeline()
