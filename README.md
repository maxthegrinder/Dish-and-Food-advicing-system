# Food Recommendation and Classification System
# System Rekomendacji i Klasyfikacji Potraw

## English

### Overview
This is a student project focusing on building a multimodal machine learning system for food classification and recommendation. The project demonstrates a solid understanding of data processing, Natural Language Processing (NLP), Computer Vision (CV), and Multimodal Deep Learning.

### Project Description
This project implements an end-to-end machine learning pipeline that identifies the cuisine of a dish based on its ingredients and a corresponding image. The goal is to build an intelligent food recommendation and classification system that processes both textual and visual information.

Key implementations in the project include:
- **Dynamic Dataset Retrieval:** Automatically fetching the latest dataset from Kaggle using `kagglehub`.
- **Text Processing:** Cleaning raw ingredient text (removing numbers, units, punctuation, and stopwords) to prepare it for natural language models.
- **Image Processing Optimization:** Implementing `tf.data.Dataset` pipelines to load and preprocess high-resolution images efficiently in batches, preventing memory exhaustion (`MemoryError`).
- **Feature Extraction & Classification:** Using TF-IDF, Word Embeddings, and pre-trained Deep Learning CNNs (ResNet50, VGG16) to extract meaningful patterns from data.
- **Multimodal Learning:** Creating a combined neural network architecture that fuses dense textual features with global image features to make a final prediction.
- **Model Explainability:** Using Grad-CAM (Gradient-weighted Class Activation Mapping) to visualize which parts of an image the model considers most important for its prediction.

### Dataset
The project uses the **Food Ingredients and Recipe Dataset with Images** from Kaggle.
The dataset contains thousands of recipes, including their titles, ingredients, cooking instructions, and corresponding images.

During runtime, the dataset is automatically downloaded via the `kagglehub` library.

### Key Features
- **Data Cleaning and Preprocessing:** Extracting clean ingredients using NLTK and formatting images using TensorFlow's `tf.data.Dataset` pipeline.
- **NLP Models:**
  - **TF-IDF + MLP:** Text classification using Term Frequency-Inverse Document Frequency and a Multi-Layer Perceptron.
  - **Embeddings + MLP:** Word embeddings combined with an MLP for ingredient analysis.
- **Computer Vision Models:**
  - **ResNet50:** A Convolutional Neural Network (CNN) pre-trained on ImageNet, fine-tuned for food image classification.
  - **VGG16:** Another pre-trained CNN for robust feature extraction and classification.
- **Multimodal Architecture:**
  - A combined model that accepts both textual data (ingredients) and image data simultaneously, concatenating their feature vectors to predict the cuisine type.
- **Explainability (Grad-CAM):** Implementation of Gradient-weighted Class Activation Mapping to visualize which parts of the food images the ResNet model focuses on when making predictions.

### How to Run
1. Ensure you have the required dependencies installed:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow opencv-python kagglehub jupyter
   ```
2. Open `Projekt.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells. The dataset will be downloaded automatically. The script will train the models in sequence and display the evaluation metrics and visualizations.

### Authors
- Maksym Volchanskyi
- Vadym Masliuk

---

## Polski

### Przegląd projektu
Jest to projekt studencki, którego celem jest zbudowanie multimodalnego systemu uczenia maszynowego do klasyfikacji i rekomendacji potraw. Projekt demonstruje solidne zrozumienie przetwarzania danych, Przetwarzania Języka Naturalnego (NLP), Widzenia Komputerowego (CV) oraz Multimodalnego Głębokiego Uczenia.

### Opis projektu
Projekt ten implementuje kompleksowy proces uczenia maszynowego (end-to-end pipeline), który identyfikuje kuchnię dania na podstawie jego składników i odpowiadającego mu zdjęcia. Celem jest zbudowanie inteligentnego systemu rekomendacji i klasyfikacji żywności, który przetwarza zarówno informacje tekstowe, jak i wizualne.

Główne implementacje w projekcie obejmują:
- **Dynamiczne pobieranie danych:** Automatyczne pobieranie najnowszego zestawu danych z Kaggle przy użyciu `kagglehub`.
- **Przetwarzanie tekstu:** Oczyszczanie surowego tekstu składników (usuwanie liczb, jednostek, znaków przestankowych i tzw. stopwords), aby przygotować je dla modeli języka naturalnego.
- **Optymalizacja przetwarzania obrazów:** Wdrożenie potoków `tf.data.Dataset` do wydajnego ładowania i wstępnego przetwarzania obrazów w wysokiej rozdzielczości w partiach, co zapobiega wyczerpaniu pamięci (`MemoryError`).
- **Ekstrakcja cech i klasyfikacja:** Wykorzystanie TF-IDF, zanurzeń słów (Word Embeddings) oraz wstępnie wytrenowanych głębokich sieci CNN (ResNet50, VGG16) do wyodrębnienia istotnych wzorców z danych.
- **Uczenie multimodalne:** Stworzenie połączonej architektury sieci neuronowej, która łączy gęste cechy tekstowe z globalnymi cechami obrazu, aby dokonać ostatecznej predykcji.
- **Wyjaśnialność modeli:** Wykorzystanie Grad-CAM (Gradient-weighted Class Activation Mapping) do wizualizacji tego, które części obrazu model uważa za najważniejsze przy podejmowaniu decyzji.

### Zbiór danych
Projekt wykorzystuje **Food Ingredients and Recipe Dataset with Images** z serwisu Kaggle.
Zbiór ten zawiera tysiące przepisów, w tym ich tytuły, składniki, instrukcje przygotowania oraz odpowiadające im obrazy.

Podczas uruchamiania, zestaw danych jest automatycznie pobierany przy użyciu biblioteki `kagglehub`.

### Główne funkcje
- **Czyszczenie i wstępne przetwarzanie danych:** Wyodrębnianie czystych składników przy użyciu NLTK oraz formatowanie obrazów przy użyciu potoku `tf.data.Dataset` z biblioteki TensorFlow.
- **Modele NLP:**
  - **TF-IDF + MLP:** Klasyfikacja tekstu z wykorzystaniem częstotliwości terminów (TF-IDF) oraz wielowarstwowego perceptronu (MLP).
  - **Zanurzenia słów (Embeddings) + MLP:** Połączenie zanurzeń słów z MLP do analizy składników.
- **Modele widzenia komputerowego (CV):**
  - **ResNet50:** Konwolucyjna sieć neuronowa (CNN) wstępnie wytrenowana na zbiorze ImageNet, dotrenowana do klasyfikacji obrazów jedzenia.
  - **VGG16:** Kolejna wstępnie wytrenowana sieć CNN służąca do solidnej ekstrakcji cech i klasyfikacji.
- **Architektura multimodalna:**
  - Połączony model, który przyjmuje jednocześnie dane tekstowe (składniki) oraz dane obrazowe, łącząc ich wektory cech do przewidywania typu kuchni.
- **Wyjaśnialność modeli (Grad-CAM):** Implementacja map aktywacji klas ważonych gradientem, aby zwizualizować, na których częściach obrazów skupia się model ResNet podczas dokonywania predykcji.

### Jak uruchomić
1. Upewnij się, że masz zainstalowane wymagane pakiety:
   ```bash
   pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow opencv-python kagglehub jupyter
   ```
2. Otwórz plik `Projekt.ipynb` w Jupyter Notebook lub JupyterLab.
3. Uruchom wszystkie komórki. Zbiór danych zostanie pobrany automatycznie. Skrypt po kolei wytrenuje modele, a następnie wyświetli metryki ewaluacyjne oraz wizualizacje.

### Autorzy
- Maksym Volchanskyi
- Vadym Masliuk
