from abc import ABC, abstractmethod
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

class BaseModel(ABC):
    """
    Abstract Base Class defining the standard interface for all models.
    Demonstrates Open-Closed Principle (OCP) and Liskov Substitution Principle (LSP).
    """
    def __init__(self, num_classes: int, learning_rate: float = 0.001):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self._build()

    @abstractmethod
    def _build(self) -> None:
        """Constructs the model architecture. Must be implemented by subclasses."""
        pass

    def train(self, x_train, y_train, x_val, y_val, epochs=20, batch_size=32, patience=5) -> dict:
        """Trains the model with early stopping."""
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        print(f"Training {self.__class__.__name__}...")
        history = self.model.fit(
            x=x_train, y=y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        return history.history

    def predict(self, x_test):
        """Generates predictions for the given data."""
        return self.model.predict(x_test)

    def evaluate(self, x_test, y_test):
        """Evaluates model performance."""
        return self.model.evaluate(x_test, y_test, verbose=0)
