import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data_preprocessor import DataPreprocessor
import os

# Dynamically get the number of logical CPU cores
available_cores = os.cpu_count()

# Set LOKY_MAX_CPU_COUNT to the number of logical cores
if available_cores:
    os.environ['LOKY_MAX_CPU_COUNT'] = str(available_cores)
else:
    # Fallback if the core count could not be determined
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Default to 4

# Load dataset
df = pd.read_csv("AppGallery.csv")


# Rename columns
df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]

# Use y2, y3, and y4 as target columns
df["y"] = df["y2"]

# Remove rows with missing or empty target values
df = df.loc[(df["y"] != '') & (~df["y"].isna())]

# Instantiate the preprocessor
preprocessor = DataPreprocessor(max_features=102)

# Preprocess the data
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["Type 2", "Type 3", "Type 4"]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define and train the model
classifier = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=2000, random_state=0))
classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test)

# Display results
# Display per-label accuracies
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
preprocessor.save_vectorizer("tfidf_vectorizer.pkl")
joblib.dump(classifier, "HDBCModel.pkl")
