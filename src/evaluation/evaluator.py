import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

class ModelEvaluator:
    """
    Evaluates predictions and calculates performance metrics.
    Demonstrates SRP by separating evaluation logic from models.
    """
    def __init__(self, class_names):
        self.class_names = class_names

    def evaluate(self, model, test_data, y_test, model_name: str):
        print(f"\n--- Evaluation for {model_name} ---")

        # Determine if test_data is a tf.data.Dataset or numpy arrays
        y_pred = model.predict(test_data)
        y_pred_classes = np.argmax(y_pred, axis=1)

        accuracy = accuracy_score(y_test, y_pred_classes)
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.class_names))

        return {
            'accuracy': accuracy,
            'f1_score': f1
        }
