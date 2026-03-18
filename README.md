# Food Recommendation and Classification System
# System Rekomendacji i Klasyfikacji Potraw

## English

### Overview
This is a student project focusing on building a multimodal machine learning system for food classification and recommendation. The project demonstrates a solid understanding of data processing, Natural Language Processing (NLP), Computer Vision (CV), and Multimodal Deep Learning.

### Software Architecture (SOLID & OOP)
This project has been heavily refactored from a procedural Jupyter Notebook into a robust, object-oriented Python package following **SOLID principles**:
- **Single Responsibility Principle (SRP):** Data loading (`KaggleDataLoader`), preprocessing (`DataPreprocessor`), dataset building (`DatasetBuilder`), and evaluation (`ModelEvaluator`) are completely decoupled into their own classes.
- **Open/Closed Principle (OCP) & Liskov Substitution Principle (LSP):** All models (`TfIdfModel`, `ResNet50Model`, `MultimodalModel`, etc.) inherit from an abstract `BaseModel`. Adding a new architecture only requires creating a new subclass without modifying the training loops.
- **Dependency Inversion Principle (DIP):** The `main.py` entry point acts as an orchestrator, injecting processed data and configuration into the loosely coupled model and evaluation modules.

### Directory Structure
```
.
├── main.py                     # Pipeline orchestrator
├── Projekt.ipynb               # Presentation Notebook
├── README.md                   # Documentation
└── src
    ├── data
    │   ├── data_loader.py      # Kaggle Hub downloading & CSV loading
    │   ├── dataset_builder.py  # tf.data.Dataset pipelines
    │   └── preprocessor.py     # NLTK text cleaning & Label Encoding
    ├── evaluation
    │   ├── evaluator.py        # Accuracy, F1 scores, Classification Reports
    │   └── visualizer.py       # Grad-CAM and Matplotlib training history
    └── models
        ├── base_model.py       # Abstract Base Class
        ├── multimodal.py       # NLP + CV fused neural network
        ├── text_models.py      # TF-IDF & Embedding MLP Models
        └── vision_models.py    # ResNet50 & VGG16 Transfer Learning Models
```

### Dataset
The project uses the **Food Ingredients and Recipe Dataset with Images** from Kaggle. During runtime, the dataset is automatically downloaded via the `kagglehub` library.

### How to Run
1. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow opencv-python kagglehub jupyter
   ```
2. You can run the entire pipeline directly from the terminal:
   ```bash
   python main.py
   ```
3. Alternatively, open `Projekt.ipynb` in Jupyter Notebook to run the pipeline interactively.

### Authors
- Maksym Volchanskyi
- Vadym Masliuk

---

## Polski

### Przegląd projektu
Jest to projekt studencki, którego celem jest zbudowanie multimodalnego systemu uczenia maszynowego do klasyfikacji i rekomendacji potraw. Projekt demonstruje solidne zrozumienie przetwarzania danych, Przetwarzania Języka Naturalnego (NLP), Widzenia Komputerowego (CV) oraz Multimodalnego Głębokiego Uczenia.

### Architektura Oprogramowania (SOLID & OOP)
Projekt został znacząco zrefaktoryzowany z proceduralnego notatnika Jupyter do solidnego, obiektowego pakietu Python, zgodnego z **zasadami SOLID**:
- **Zasada pojedynczej odpowiedzialności (SRP):** Pobieranie danych (`KaggleDataLoader`), ich wstępne przetwarzanie (`DataPreprocessor`), budowanie strumieni danych (`DatasetBuilder`) i ewaluacja (`ModelEvaluator`) są całkowicie odseparowane do oddzielnych klas.
- **Zasada otwarte-zamknięte (OCP) i Zasada podstawienia Liskov (LSP):** Wszystkie modele (`TfIdfModel`, `ResNet50Model`, `MultimodalModel` itd.) dziedziczą po abstrakcyjnej klasie bazowej `BaseModel`. Dodanie nowej architektury wymaga jedynie stworzenia nowej subklasy, bez konieczności modyfikowania pętli uczących.
- **Zasada odwrócenia zależności (DIP):** Plik wejściowy `main.py` pełni rolę orkiestratora, wstrzykując przetworzone dane oraz konfiguracje do luźno powiązanych modułów modeli i ewaluacji.

### Struktura Katalogów
```
.
├── main.py                     # Orkiestrator potoku
├── Projekt.ipynb               # Notatnik prezentacyjny
├── README.md                   # Dokumentacja
└── src
    ├── data
    │   ├── data_loader.py      # Pobieranie z Kaggle Hub i wczytywanie CSV
    │   ├── dataset_builder.py  # Potoki tf.data.Dataset
    │   └── preprocessor.py     # Czyszczenie tekstu (NLTK) i Kodowanie Etykiet
    ├── evaluation
    │   ├── evaluator.py        # Dokładność, F1, Raporty Klasyfikacji
    │   └── visualizer.py       # Historia uczenia oraz Grad-CAM
    └── models
        ├── base_model.py       # Abstrakcyjna Klasa Bazowa
        ├── multimodal.py       # Sieć połączona (NLP + CV)
        ├── text_models.py      # Modele MLP (TF-IDF & Embeddings)
        └── vision_models.py    # Modele ResNet50 & VGG16
```

### Zbiór danych
Projekt wykorzystuje **Food Ingredients and Recipe Dataset with Images** z serwisu Kaggle. Podczas uruchamiania zestaw danych jest pobierany automatycznie.

### Jak uruchomić
1. Zainstaluj wymagane pakiety:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow opencv-python kagglehub jupyter
   ```
2. Uruchom główny potok z terminala:
   ```bash
   python main.py
   ```
3. Alternatywnie, uruchom plik `Projekt.ipynb` w środowisku Jupyter.

### Autorzy
- Maksym Volchanskyi
- Vadym Masliuk
