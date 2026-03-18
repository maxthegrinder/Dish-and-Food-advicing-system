import ast
import re
import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import cv2

class DataPreprocessor:
    """
    Handles preprocessing text, labels, and filtering missing images.
    Demonstrates SRP by isolating cleaning and transformations.
    """
    def __init__(self, df: pd.DataFrame, images_dir: str):
        self.df = df
        self.images_dir = images_dir
        self.label_encoder = LabelEncoder()

        # Ensure NLTK stopwords are downloaded
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def generate_missing_cuisines(self):
        """Generates a pseudo-Cuisine column if one doesn't exist."""
        if 'Cuisine' not in self.df.columns:
            print("Creating pseudo-Cuisine column for demonstration...")
            def assign_cuisine(title):
                title = str(title).lower()
                if any(word in title for word in ['pasta', 'pizza', 'italian', 'lasagna']):
                    return 'Italian'
                if any(word in title for word in ['taco', 'mexican', 'enchilada', 'burrito']):
                    return 'Mexican'
                if any(word in title for word in ['curry', 'indian', 'tikka', 'masala']):
                    return 'Indian'
                if any(word in title for word in ['soy', 'asian', 'stir-fry', 'chinese', 'japanese']):
                    return 'Asian'
                return 'American'

            self.df['Cuisine'] = self.df['Title'].apply(assign_cuisine)
        return self.df

    def remove_missing_images(self):
        """Filters out rows where the image file does not exist on disk or is corrupt."""
        valid_indices = []
        for idx, row in self.df.iterrows():
            image_name = row['Image_Name']
            if pd.isna(image_name):
                continue
            image_path = os.path.join(self.images_dir, str(image_name) + '.jpg')

            # Check if file exists and is a valid readable image
            if os.path.exists(image_path):
                # Optionally use cv2 to quickly verify the image header to prevent TensorFlow crash
                # In larger datasets we might do this, but to be safe:
                # We just ensure it's not a 0-byte file
                if os.path.getsize(image_path) > 0:
                    valid_indices.append(idx)

        print(f"Removed {len(self.df) - len(valid_indices)} rows with missing or empty images.")
        self.df = self.df.loc[valid_indices].reset_index(drop=True)
        self.df['Image_Path'] = self.df['Image_Name'].apply(
            lambda x: os.path.join(self.images_dir, str(x) + '.jpg')
        )
        return self.df

    def encode_labels(self):
        """Fits the label encoder and creates a new column for numeric labels."""
        self.df['Cuisine_Label'] = self.label_encoder.fit_transform(self.df['Cuisine'])
        return self.df['Cuisine_Label']

    def get_num_classes(self) -> int:
        return len(self.label_encoder.classes_)

    def get_class_names(self):
        return self.label_encoder.classes_

    def _clean_ingredients_text(self, ingredients_list):
        """Helper method to clean a single ingredient list."""
        if isinstance(ingredients_list, str):
            try:
                ingredients_list = ast.literal_eval(ingredients_list)
            except (ValueError, SyntaxError):
                ingredients_list = [ingredients_list]

        ingredients_text = ' '.join(ingredients_list)
        ingredients_text = re.sub(r'\d+', '', ingredients_text)
        ingredients_text = re.sub(r'cup|tablespoon|teaspoon|pound|ounce|tbsp|tsp|oz|lb|g|ml|l', '', ingredients_text, flags=re.IGNORECASE)
        ingredients_text = re.sub(r'[^\w\s]', '', ingredients_text)
        ingredients_text = ingredients_text.lower()

        words = ingredients_text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)

    def preprocess_text_features(self, max_tfidf_features=1000, num_words_tokenizer=5000, max_seq_length=100):
        """
        Cleans text and extracts both TF-IDF features and Padded Sequences for embeddings.
        Returns the data representations and the Tokenizer.
        """
        print("Preprocessing text features...")
        self.df['Processed_Ingredients'] = self.df['Cleaned_Ingredients'].apply(self._clean_ingredients_text)

        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer(max_features=max_tfidf_features)
        X_tfidf_sparse = tfidf_vectorizer.fit_transform(self.df['Processed_Ingredients'])
        X_tfidf_dense = X_tfidf_sparse.toarray()

        # Tokenizer
        tokenizer = Tokenizer(num_words=num_words_tokenizer)
        tokenizer.fit_on_texts(self.df['Processed_Ingredients'])
        X_seq = tokenizer.texts_to_sequences(self.df['Processed_Ingredients'])
        X_padded = pad_sequences(X_seq, maxlen=max_seq_length)

        return X_tfidf_dense, X_padded, tokenizer
