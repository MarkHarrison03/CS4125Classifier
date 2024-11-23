import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from data_preprocessor.data_preprocessor import DataPreprocessor

print("Loading dataset...")
df = pd.read_csv("../../AppGallery.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

preprocessor = DataPreprocessor(max_features=2000)

print("Preprocessing dataset...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 2", "Type 3", "Type 4"]
)
print(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}\n")

print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print("Training the Naive Bayes model wrapped with MultiOutputClassifier...")
nb_classifier = MultiOutputClassifier(MultinomialNB())
nb_classifier.fit(X_train, y_train)
print("Naive Bayes model training completed.\n")

print("Evaluating the Naive Bayes model...")
y_pred = nb_classifier.predict(X_test)

print("\nClassification Reports for each label:")
for i, label in enumerate(["Type 2", "Type 3", "Type 4"]):
    print(f"\nClassification Report for {label}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))

print("\nSaving the Naive Bayes model...")
os.makedirs("./exported_models/NB", exist_ok=True)
preprocessor.save_vectorizer("./exported_models/NB/nb_tfidf_vectorizer.pkl")
joblib.dump(nb_classifier, "./exported_models/NB/NaiveBayesModel.pkl")
print("Naive Bayes model saved successfully.")
