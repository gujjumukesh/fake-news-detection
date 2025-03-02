from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

app = Flask(__name__, static_folder='static')

# Load the dataset
df = pd.read_csv(r"C:\Users\g.mukesh\Downloads\bbc\BBCNews.csv")

# Data preprocessing
def preprocess_text(text):
    text = text.lower()  # Lowercase the text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['text'] = df['text'].apply(preprocess_text)  # Apply preprocessing to the text column

# Split the data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
vectorizer = TfidfVectorizer()
X_train_std = vectorizer.fit_transform(X_train)
X_test_std = vectorizer.transform(X_test)

# Model training with class weight to handle imbalance
logi = LogisticRegression(class_weight='balanced')  # Adds class weight for balancing
logi.fit(X_train_std, y_train)

# Evaluate model performance on the test set
y_pred = logi.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(conf_matrix)

# Prediction function with threshold adjustment
def predict_news(news):
    news_std = vectorizer.transform([news])  # Vectorize the input news article
    prediction_prob = logi.predict_proba(news_std)[0]  # Get the probabilities for both classes
    print(f"Probabilities: Fake News={prediction_prob[0]}, True News={prediction_prob[1]}")

    # Adjust the threshold from 0.5 to something higher if needed (e.g., 0.6)
    threshold = 0.2  # Modify this threshold to make the model more conservative
    if prediction_prob[1] >= threshold:  # Probability of class 1 (True News)
        return "True News"
    else:
        return "Fake News"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        news_article = request.form['news']
        news_article = preprocess_text(news_article)
        prediction = predict_news(news_article)
        return render_template('news.html', prediction=prediction, news_article=news_article)
    return render_template('news.html')

if __name__ == '__main__':
    app.run(debug=True)
