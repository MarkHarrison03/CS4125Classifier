import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_preprocessor.data_preprocessor import DataPreprocessor

required_columns = ["Interaction content", "Ticket Summary", "Type 1", "Type 2", "Type 3", "Type 4"]

preprocessor = DataPreprocessor(max_features=2000)

print("Preprocessing datasets...")
df = preprocessor.preprocess_datasets(
    required_columns=["Interaction content", "Ticket Summary", "Type 1", "Type 2", "Type 3", "Type 4"],
    translate=True
)


print("Preprocessing text and extracting features...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 1", "Type 2", "Type 3", "Type 4"]
)
print(f"Preprocessing completed.\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
print(f"Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}\n")

print("Training the SVM model...")
svm_model = MultiOutputClassifier(SVC(kernel="linear", probability=True, random_state=42))
svm_model.fit(X_train, y_train)
print("SVM model training completed.\n")

print("Evaluating the SVM model...")
y_pred = svm_model.predict(X_test)

accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracies.append(accuracy)

print("Per-label accuracies:", accuracies)
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy: {average_accuracy:.2f}")

for i, label in enumerate(["Type 1", "Type 2", "Type 3", "Type 4"]):
    print(f"\nClassification Report for {label}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

print("\nSaving the SVM model and vectorizer...")
os.makedirs("./exported_models/SVM", exist_ok=True)
preprocessor.save_vectorizer("./exported_models/SVM/svm_tfidf_vectorizer.pkl")
joblib.dump(svm_model, "./exported_models/SVM/SVMModel.pkl")
print("SVM model and vectorizer saved successfully.\n")

print("Script execution completed successfully.")
