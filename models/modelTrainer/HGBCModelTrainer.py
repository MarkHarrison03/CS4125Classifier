import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessor.data_preprocessor import DataPreprocessor

import warnings
# Set LOKY_MAX_CPU_COUNT dynamically
available_cores = os.cpu_count()
os.environ['LOKY_MAX_CPU_COUNT'] = str(available_cores if available_cores else 4)

# Suppress loky backend warnings
warnings.filterwarnings("ignore", message="Could not find the number of physical cores")

# Load dataset
print("Loading dataset...")
df = pd.read_csv("AppGallery.csv")
print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# Instantiate the preprocessor
preprocessor = DataPreprocessor(max_features=102)

# Prepare target columns and filter data
print("Preparing target columns and filtering data...")
target_columns = ["Type 1", "Type 2", "Type 3", "Type 4"]
df = preprocessor.prepare_targets(df, target_columns, main_target="Type 2")
print(f"Target preparation completed. Remaining rows: {df.shape[0]}\n")

# Preprocess the data
print("Preprocessing dataset...")
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 2", "Type 3", "Type 4"]
)
print(f"Preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}\n")

# Train/test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"Data split completed. Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}\n")

# Define and train the model
print("Training the model...")
classifier = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=2000, random_state=0))
classifier.fit(X_train, y_train)
print("Model training completed.\n")

# Predict on the test set
print("Evaluating the model...")
y_pred = classifier.predict(X_test)

# Display results
accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracies.append(accuracy)

print("Per-label accuracies:", accuracies)
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy: {average_accuracy:.2f}")

# Classification reports for each label
for i, label in enumerate(["Type 2", "Type 3", "Type 4"]):
    print(f"Classification Report for {label}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    print("\n")

# Save the vectorizer and model
print("Saving the vectorizer and model...")
preprocessor.save_vectorizer("exported_models/HGBC/tfidf_vectorizer.pkl")
joblib.dump(classifier, "exported_models/HGBC/HGBCModel.pkl")
print("Model and vectorizer saved successfully.")
