import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from data_preprocessor import DataPreprocessor  # Assuming the preprocessor is saved in `data_preprocessor.py`

# Load the dataset
df = pd.read_csv("AppGallery.csv")

# Convert dtype object to unicode string
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

# Optional: Rename variable names for readability
df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]

# Use y2 as the target
df["y"] = df["y2"]

# Remove empty y rows
df = df.loc[(df["y"] != '') & (~df["y"].isna())]
print(f"Dataset shape after filtering: {df.shape}")

# Instantiate the preprocessor
preprocessor = DataPreprocessor(max_features=2000)

# Preprocess the dataset
X, y = preprocessor.preprocess_dataframe(
    df,
    content_col="Interaction content",
    summary_col="Ticket Summary",
    label_cols=["y"]
)

# Convert labels to a numpy array
y = y.to_numpy().flatten()

# Filter bad test cases
y_series = pd.Series(y)
good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index
y_good = y[y_series.isin(good_y_value)]
X_good = X[y_series.isin(good_y_value)]
y_bad = y[y_series.isin(good_y_value) == False]
X_bad = X[y_series.isin(good_y_value) == False]

# Adjust the test size based on the good cases
test_size = X.shape[0] * 0.2 / X_good.shape[0]
print(f"Adjusted test size: {test_size}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=test_size, random_state=0)

# Add back the bad test cases to the training set
X_train = np.concatenate((X_train, X_bad), axis=0)
y_train = np.concatenate((y_train, y_bad), axis=0)

# Model selection
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)

# Training
classifier.fit(X_train, y_train)

# Testing
y_pred = classifier.predict(X_test)

# Display results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the preprocessor and model
preprocessor.save_vectorizer("tfidf_vectorizer.pkl")
import joblib
joblib.dump(classifier, "RandomForestModel.pkl")
