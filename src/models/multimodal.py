import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate, Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from src.models.vision_models import PreTrainedVisionModel

class MultimodalModel(PreTrainedVisionModel):
    """
    Multimodal neural network fusing text and image embeddings.
    Inherits training logic from PreTrainedVisionModel for tf.data.Datasets.
    """
    def __init__(self, text_input_dim: int, image_input_shape: tuple, num_classes: int, learning_rate: float = 0.001):
        self.text_input_dim = text_input_dim
        self.image_input_shape = image_input_shape
        super().__init__(num_classes, learning_rate)

    def _build(self) -> None:
        # Gałąź tekstowa (MLP)
        text_input = Input(shape=(self.text_input_dim,), name="text_input")
        text_features = Dense(256, activation='relu')(text_input)
        text_features = Dropout(0.3)(text_features)
        text_features = Dense(128, activation='relu')(text_features)

        # Gałąź obrazowa (CNN)
        image_input = Input(shape=self.image_input_shape, name="image_input")
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=image_input)

        # Zamrożenie warstw bazowego modelu
        for layer in base_model.layers:
            layer.trainable = False

        image_features = GlobalAveragePooling2D()(base_model.output)
        image_features = Dense(256, activation='relu')(image_features)
        image_features = Dropout(0.3)(image_features)

        # Połączenie gałęzi
        combined = concatenate([text_features, image_features])

        # Wspólne warstwy
        x = Dense(256, activation='relu')(combined)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)

        # Warstwa wyjściowa
        output = Dense(self.num_classes, activation='softmax')(x)

        # Tworzenie modelu
        model = Model(inputs=[text_input, image_input], outputs=output)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        self.model = model
