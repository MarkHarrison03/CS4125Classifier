import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_preprocessor.data_preprocessor import DataPreprocessor

print("Loading dataset...")
df = pd.read_csv("./AppGallery.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

preprocessor = DataPreprocessor(max_features=2000)

print("Preprocessing dataset...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 2"]
)
print(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training the k-Nearest Neighbors model...")
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train.values.ravel())
print("k-Nearest Neighbors model training completed.\n")

print("Evaluating the k-Nearest Neighbors model...")
y_pred = knn_classifier.predict(X_test)

print("\nConfusion Matrix (k-Nearest Neighbors):")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report (k-Nearest Neighbors):")
print(classification_report(y_test, y_pred))

print("\nSaving the k-Nearest Neighbors model...")
os.makedirs("./exported_models/KNN", exist_ok=True)
preprocessor.save_vectorizer("./exported_models/KNN/nn_tfidf_vectorizer.pkl")
joblib.dump(knn_classifier, "./exported_models/KNN/KNNModel.pkl")
print("k-Nearest Neighbors model saved successfully.")
