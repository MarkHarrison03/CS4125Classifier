import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from data_preprocessor import DataPreprocessor  # Updated preprocessor with `prepare_targets`

# Load the dataset
print("Loading dataset...")
df = pd.read_csv("AppGallery.csv")
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

# Model selection
print("Training the Random Forest model...")
classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)
print("Model training completed.\n")

# Testing
print("Evaluating the model...")
y_pred = classifier.predict(X_test)

# Display results
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the preprocessor and model
print("Saving the vectorizer and model...")
preprocessor.save_vectorizer("tfidf_vectorizer.pkl")
joblib.dump(classifier, "RandomForestModel.pkl")
print("Model and vectorizer saved successfully.")
