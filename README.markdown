# Fake Reviews Detection Project

## Overview
This project implements a machine learning pipeline to detect fake product reviews using the `fake reviews dataset.csv`. It is part of the Machine Learning course (IIA4, 2024/2025). The pipeline includes data preprocessing, exploratory data analysis (EDA), text vectorization, modeling, evaluation, and interpretation using LIME. Two models are implemented: Logistic Regression and MLP Classifier. The project also answers theoretical questions about text processing, evaluation metrics, and ethical considerations.

## Dataset
The dataset (`fake reviews dataset.csv`) contains product reviews labeled as:
- **CG**: Computer-generated (fake)
- **OR**: Original (real)

**Columns**:
- `category`: Product category
- `rating`: Review rating (1–5)
- `label`: CG (fake) or OR (real)
- `text_`: Review text

## Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Required Python packages (listed below)

## Installation
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud lime
   ```
3. Download NLTK resources:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
4. Ensure the dataset (`fake reviews dataset.csv`) is in the same directory as the notebook.

## Usage
1. Open the Jupyter Notebook (`fake_reviews_detection.ipynb`) in Jupyter Notebook or JupyterLab.
2. Run all cells sequentially to:
   - Load and preprocess the dataset
   - Perform EDA (visualizations: class distribution, review length, WordClouds, bigrams)
   - Vectorize text using TF-IDF
   - Train and evaluate Logistic Regression and MLP Classifier models
   - Interpret predictions using LIME
   - View answers to theoretical questions
3. Outputs (plots, metrics, and explanations) will be displayed inline.

## Project Structure
- `fake_reviews_detection.ipynb`: Main Jupyter Notebook with the complete pipeline.
- `fake reviews dataset.csv`: Input dataset (not included in the repository; ensure you have it).
- `README.md`: This file.

## Key Features
- **Preprocessing**: Text cleaning (lowercase, punctuation removal, stopword removal, lemmatization).
- **EDA**: Visualizations to understand class distribution, review length, and word patterns.
- **Modeling**: Logistic Regression (baseline) and MLP Classifier (neural network).
- **Evaluation**: Accuracy, precision, recall, F1-score, AUC, ROC curves, and confusion matrices.
- **Interpretation**: LIME explanations for two reviews to highlight influential words.
- **Theoretical Answers**: Markdown cells addressing text cleaning, F1-score, TF-IDF vs. Word2Vec, class imbalance, and ethical issues.

## Results
- **Model Performance**: Logistic Regression typically achieves high accuracy and F1-score, with MLP Classifier offering comparable results.
- **Insights**: Fake reviews may use repetitive phrases, detectable via bigram analysis and LIME.
- **Limitations**: Potential class imbalance and dataset size may affect generalization.

## Notes
- Ensure sufficient memory for TF-IDF vectorization (5000 features). Reduce `max_features` if needed.
- The dataset is assumed to be balanced; EDA will confirm this.
- For advanced analysis, consider incorporating the `category` feature or experimenting with embeddings (e.g., Word2Vec).

## Ethical Considerations
- **False Positives**: Misclassifying real reviews as fake can harm users or businesses.
- **Bias**: Models may inherit biases from the dataset.
- **Privacy**: Review data may contain sensitive information.
- **Transparency**: Interpretable models (e.g., LIME) are crucial for trust.

## Author
- Created for the Machine Learning course (IIA4, 2024/2025).
- Contact: [NAJAR Yassine]

## License
This project is for educational purposes only. Ensure compliance with dataset usage terms.
