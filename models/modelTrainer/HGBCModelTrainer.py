import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_preprocessor.data_preprocessor import DataPreprocessor

print("Loading dataset...")
df = pd.read_csv("../../AppGallery.csv")
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
print(f"Data split completed. Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}\n")

print("Training the HistGradientBoostingClassifier model...")
classifier = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=5000, random_state=0))
classifier.fit(X_train, y_train)
print("Model training completed.\n")

print("Evaluating the model...")
y_pred = classifier.predict(X_test)

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

print("\nSaving the model and vectorizer...")
os.makedirs("./exported_models/HGBC", exist_ok=True)
preprocessor.save_vectorizer("./exported_models/HGBC/tfidf_vectorizer.pkl")
joblib.dump(classifier, "./exported_models/HGBC/HGBCModel.pkl")
print("HGBC model saved successfully.")
