import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

# Load the training data
train_data = pd.read_csv('train.csv')

# Split the data into features (X) and target labels (y)
X = train_data['text']
y = train_data['target']

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the validation data
X_valid_tfidf = vectorizer.transform(X_valid)

# Initialize a Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Predict on the validation data
y_pred = model.predict(X_valid_tfidf)

# Calculate the F1 score
f1 = f1_score(y_valid, y_pred)

# Print the F1 score
print("Validation F1 Score:", f1)
