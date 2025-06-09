# 🎬 Sentiment Analysis on IMDB Movie Reviews

This mini project performs sentiment analysis on a dataset of IMDB movie reviews using Natural Language Processing (NLP) and machine learning classifiers (Decision Tree and Random Forest). It includes both a data processing pipeline and a user-friendly interface powered by **Gradio**.

---

## 📁 Project Structure

- `mp_2.py`: Main Python script for data cleaning, model training, accuracy comparison, and Gradio app launch.
- `mp_doc.ipynb`: Jupyter Notebook version of the project for explanation and step-by-step visualization.
- `IMDB Dataset.csv.zip`: Compressed dataset containing IMDB reviews with sentiment labels.

---

## 🧠 Features

- Preprocessing of text: HTML tag removal, lowercasing, stopword removal, stemming
- Text vectorization using `CountVectorizer`
- Classification using:
  - Decision Tree
  - Random Forest
- Accuracy comparison using bar and line charts
- Live sentiment prediction interface with **Gradio**

---

## 🚀 How to Run

### 1. Install Dependencies

Make sure the following Python libraries are installed:

```bash
pip install pandas matplotlib nltk scikit-learn gradio

