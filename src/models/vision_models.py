import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from src.models.base_model import BaseModel

class PreTrainedVisionModel(BaseModel):
    """
    Base class for Transfer Learning vision models.
    """
    def __init__(self, num_classes: int, learning_rate: float = 0.001):
        super().__init__(num_classes, learning_rate)

    def _build(self) -> None:
        pass # Implemented by child classes

    def train(self, train_dataset, val_dataset, epochs=10, patience=5) -> dict:
        """Overridden to handle tf.data.Dataset training."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        print(f"Training {self.__class__.__name__}...")
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        return history.history

class ResNet50Model(PreTrainedVisionModel):
    """
    ResNet50 based image classifier.
    """
    def _build(self) -> None:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        self.model = Sequential([
            Input(shape=(224, 224, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

class VGG16Model(PreTrainedVisionModel):
    """
    VGG16 based image classifier.
    """
    def _build(self) -> None:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        for layer in base_model.layers:
            layer.trainable = False

        self.model = Sequential([
            Input(shape=(224, 224, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(self.num_classes, activation='softmax')
        ])

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
