import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Input
from tensorflow.keras.optimizers import Adam
from src.models.base_model import BaseModel

class TfIdfModel(BaseModel):
    """
    Multi-Layer Perceptron model for TF-IDF feature classification.
    """
    def __init__(self, input_dim: int, num_classes: int, learning_rate: float = 0.001):
        self.input_dim = input_dim
        super().__init__(num_classes, learning_rate)

    def _build(self) -> None:
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model

class EmbeddingModel(BaseModel):
    """
    Embedding + MLP model for padded text sequences.
    """
    def __init__(self, vocab_size: int, embedding_dim: int, input_length: int, num_classes: int, learning_rate: float = 0.001):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        super().__init__(num_classes, learning_rate)

    def _build(self) -> None:
        model = Sequential([
            Input(shape=(self.input_length,)),
            Embedding(self.vocab_size, self.embedding_dim),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
