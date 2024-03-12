"""
Created on Wed Nov 29 11:34:40 2023

@author: lapra
"""

import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

# Loading the csv into the Python project, encoding 'utf-8-sig' for special characters
# The pandas library removes the non-printable characters present inside each comment inside of 'CONTENT'
data = pd.read_csv('Youtube02-KatyPerry.csv', encoding='utf-8-sig')

# Importing NLTK and downloading stopwords to be used for the next step
import nltk
nltk.download('stopwords')

target_data = data[['CONTENT', 'CLASS']]
stop_words = set(stopwords.words('english'))

# Printing the shape of the two targeted columns
print("Shape of the data before modifications:", target_data.shape)

# Creating a function to tokenize comments
def tokenize_comments(comment):
    tokenize_words = comment.split()
    filter_words = [tokenize_word.lower() for tokenize_word in tokenize_words if tokenize_word.lower() not in stop_words]
    return ' '.join(filter_words)

# Apply the tokenize_comments function to the 'CONTENT' column
tokenized_data = target_data['CONTENT'].apply(tokenize_comments)

# Print the tokenized data
print("Tokenized Data:")
print(tokenized_data.head(3))

# Count vectorization
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(tokenized_data)
print("Shape of the data after Count Vectorization:", count_matrix.shape)

# TF-IDF transformation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(count_matrix)
print("Shape of the data after TF-IDF transformation:", X_tfidf.shape)

# Both the shapes of the data after Count Vectorization and TF-IDF transformation 
# have the same amount of unique words

# Shuffle the dataset using .sample
# Frac = 1, means 100% data is used for sampling (TY for the email response!)
shuffled_data = data.sample(frac = 1)

# Split the dataset into 75% for training and 25% for testing
train_size = int(0.75 * len(shuffled_data))
train_data = shuffled_data[:train_size]
test_data = shuffled_data[train_size:]

# Separate the class from the features
# Training Data
X_train = count_vectorizer.fit_transform(train_data['CONTENT'].apply(tokenize_comments))
X_train_tfidf = tfidf_transformer.fit_transform(X_train)
y_train = train_data['CLASS']

# Testing Data
X_test = count_vectorizer.transform(test_data['CONTENT'].apply(tokenize_comments))
X_test_tfidf = tfidf_transformer.transform(X_test)
y_test = test_data['CLASS']

# Fit the training data into a Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tfidf, y_train)

# Cross validate the model on the training data using 5-fold
cross_val_results = cross_val_score(naive_bayes_classifier, X_train_tfidf, y_train, cv=5)
print("Cross-validation results (accuracy):", cross_val_results)
print("Mean accuracy:", cross_val_results.mean())

# Test the model on the test data
y_pred = naive_bayes_classifier.predict(X_test_tfidf)

# Print the confusion matrix and accuracy of the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy on the test data:", accuracy)

# The confusion matrix in order:
    #1. True Negative
    #2. False Possitive
    #3. False Negative
    #4. True Possitive

# Recall = TP / TP + FN
TP = conf_matrix[1, 1]
FN = conf_matrix[1, 0]
recall = TP / (TP + FN)
print("Recall:", recall)

# New comments for testing
new_comments = ["Click this link to win a free iPad!",
                       "I wish she made more music... freeiphone.com",
                       "Watching all of the wild animals is crazy!",
                       "I am not a fan of this genre of music",
                       "Free katy perry I wish she was free",
                       "I love her music."]

# Preparing the new comments for prediction by first tokenizing 
# and filtering stop words, then transforming them into a TF-IDF representation
new_comments_tfidf = tfidf_transformer.transform(count_vectorizer.transform([tokenize_comments(comment) for comment in new_comments]))

# Predict using the trained classifier
new_comments_pred = naive_bayes_classifier.predict(new_comments_tfidf)

# Display the results for new comments
results_df = pd.DataFrame({'Comment': new_comments, 'Predicted Class': new_comments_pred})
print("Predictions for New Comments:")
print(results_df)
