import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_preprocessor.data_preprocessor import DataPreprocessor

print("Loading dataset...")
df = pd.read_csv("AppGallery.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

preprocessor = DataPreprocessor(max_features=2000)

print("Preprocessing dataset...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 1", "Type 2", "Type 3", "Type 4"]
)
print(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

print("Training the k-Nearest Neighbors model wrapped with MultiOutputClassifier...")
knn_classifier = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))
knn_classifier.fit(X_train, y_train)
print("k-Nearest Neighbors model training completed.\n")

print("Evaluating the k-Nearest Neighbors model...")
y_pred = knn_classifier.predict(X_test)

for i, label in enumerate(["Type 1", "Type 2", "Type 3", "Type 4"]):
    print(f"\nClassification Report for {label}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracies.append(accuracy)

print("Per-label accuracies:", accuracies)
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy: {average_accuracy:.2f}")
print("\nSaving the k-Nearest Neighbors model...")
os.makedirs("./exported_models/KNN", exist_ok=True)
preprocessor.save_vectorizer("./exported_models/KNN/nn_tfidf_vectorizer.pkl")
joblib.dump(knn_classifier, "./exported_models/KNN/KNNModel.pkl")
print("k-Nearest Neighbors model saved successfully.")
