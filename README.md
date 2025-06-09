# 🎬 Sentiment Analysis on IMDB Movie Reviews

This mini project performs sentiment analysis on a dataset of IMDB movie reviews using Natural Language Processing (NLP) and machine learning classifiers (Decision Tree and Random Forest). It includes both a data processing pipeline and a user-friendly interface powered by **Gradio**.

## 📁 Project Structure
- `mp_2.py`: Main Python script for data cleaning, model training, accuracy comparison, and Gradio app launch.
- `mp_doc.ipynb`: Jupyter Notebook version of the project for explanation and step-by-step visualization.
- `IMDB Dataset.csv.zip`: Compressed dataset containing IMDB reviews with sentiment labels.

## 🧠 Features
- Preprocessing of text: HTML tag removal, lowercasing, stopword removal, stemming
- Text vectorization using `CountVectorizer`
- Classification using:
  - Decision Tree
  - Random Forest
- Accuracy comparison using bar and line charts
- Live sentiment prediction interface with **Gradio**

## 🚀 How to Run

### 1. Install Dependencies
Make sure the following Python libraries are installed:

pip install pandas matplotlib nltk scikit-learn gradio

Also run this once to download NLTK resources:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

### 2. Run the Main Script
python mp_2.py

This will:
- Train the models
- Show accuracy graphs
- Launch a Gradio web interface to test your own review inputs

## 🖥 Sample Gradio Output
> "The movie was absolutely wonderful and touching!"  
✅ Output: `positive`

> "It was a waste of time with terrible acting."  
❌ Output: `negative`

## 📊 Accuracy Scores (Sample Output)
| Classifier      | Accuracy |
|-----------------|----------|
| Decision Tree   | ~0.80    |
| Random Forest   | ~0.87    |

## 📌 Notes
- Dataset path in `mp_2.py` is hardcoded; change it to relative if running elsewhere.
- You can enhance this project by adding more models (like SVM or LSTM) or deploying it to Hugging Face Spaces.

## 👩‍💻 Author
Shivasri Bhavana  
GitHub: [@harishiva6753](https://github.com/harishiva6753)

## 📂 Dataset Credit
Source: [Kaggle IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## 📂 References
Mais Yasen, Sara Tedmori, “Movies Reviews Sentiment Analysis and Classification”, IEEE Jordan International Joint, 2019.

Vivek Singh, R Piryani, Ashraf Uddin and Pranav Waila, "Sentiment analysis of movie reviews: A new feature-based heuristic for aspect-level sentiment classification" International Mutli-Conference on Automation, Computing, Communication, Control and Compressed Sensing, 2013.

Palak Baid, Apoorva Gupta, Neelam Chaplot, “Sentiment Analysis of Movie Reviews using Machine Learning Techniques” International Journal of Computer Applications, 2017.

