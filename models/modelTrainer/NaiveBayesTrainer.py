# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data_preprocessor.data_preprocessor import DataPreprocessor

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("./AppGallery.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Instantiate the preprocessor
preprocessor = DataPreprocessor(max_features=2000)

# Prepare target columns and filter rows with missing values
print("Preparing target columns and filtering data...")
target_columns = ["Type 1", "Type 2", "Type 3", "Type 4"]
df = preprocessor.prepare_targets(df, target_columns, main_target="Type 2")
print(f"Target preparation completed. Remaining rows: {df.shape[0]}\n")

# Preprocess the dataset
print("Preprocessing dataset...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["y"]
)
print(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}\n")

# Convert labels to a numpy array
y = y.to_numpy().flatten()

# Filter bad test cases
print("Filtering bad test cases...")
y_series = pd.Series(y)
good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
y_good = y[y_series.isin(good_y_value)]
X_good = X[y_series.isin(good_y_value)]
y_bad = y[y_series.isin(good_y_value) == False]
X_bad = X[y_series.isin(good_y_value) == False]

# Adjust the test size based on the good cases
test_size = X.shape[0] * 0.2 / X_good.shape[0]
print(f"Adjusted test size: {test_size}\n")

# Train/test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=test_size, random_state=0)

# Add back the bad test cases to the training set
X_train = np.concatenate((X_train, X_bad), axis=0)
y_train = np.concatenate((y_train, y_bad), axis=0)
print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}\n")

# Train Naive Bayes model
print("Training the Naive Bayes model...")
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)
print("Naive Bayes model training completed.\n")

# Testing
print("Evaluating the Naive Bayes model...")
y_pred_nb = nb_classifier.predict(X_test)

# Display results
print("Confusion Matrix (Naive Bayes):")
print(confusion_matrix(y_test, y_pred_nb))
print("\nClassification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_nb))

# Save the model
print("Saving the Naive Bayes model...")
preprocessor.save_vectorizer("./exported_models/NB/nb_tfidf_vectorizer.pkl")
joblib.dump(nb_classifier, "./exported_models/NB/NaiveBayesModel.pkl")
print("Naive Bayes model saved successfully.\n")
