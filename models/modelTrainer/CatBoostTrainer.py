import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_preprocessor.data_preprocessor import DataPreprocessor

print("Loading dataset...")
df = pd.read_csv("AppGallery.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

preprocessor = DataPreprocessor(max_features=102)

print("Preprocessing dataset...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 1", "Type 2", "Type 3", "Type 4"]
)
print(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training the CatBoost model with MultiOutputClassifier...")
# Wrap cb with multioutput classifier
cb_classifier = CatBoostClassifier(
    iterations=100, depth=6, learning_rate=0.1,
    loss_function='MultiClass', verbose=False
)

multi_output_classifier = MultiOutputClassifier(cb_classifier)
multi_output_classifier.fit(X_train, y_train)
print("Model training completed.\n")

print("Evaluating the CatBoost model...")
y_pred = multi_output_classifier.predict(X_test)
y_pred = y_pred.reshape(-1, y_test.shape[1])

print(f"Shape of y_test: {y_test.shape}")
print(f"Shape of y_pred: {y_pred.shape}")

accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracies.append(accuracy)

print("\nConfusion Matrices for each output (CatBoost):")
for i in range(y_test.shape[1]):
    print(f"\nConfusion Matrix for Output {i + 1}:")
    print(confusion_matrix(y_test.iloc[:, i], y_pred[:, i]))

# Print overall accuracy
print(f"Overall accuracies for each output: {accuracies}")

# Save the model and vectorizer
print("\nSaving the CatBoost model...")
os.makedirs("./exported_models/CB", exist_ok=True)
preprocessor.save_vectorizer("./exported_models/CB/cb_tfidf_vectorizer.pkl")
joblib.dump(multi_output_classifier, "./exported_models/CB/CatBoostModel.pkl")
print("CatBoost model saved successfully.")
