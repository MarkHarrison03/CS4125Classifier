import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

# Load dataset
df = pd.read_csv("AppGallery.csv")

# Preprocess dataset
df['Interaction content'] = df['Interaction content'].astype(str)
df['Ticket Summary'] = df['Ticket Summary'].astype(str)

df["y"] = df["Type 2"]
labels = df[['Type 2', 'Type 3', 'Type 4']]

# Remove empty target values
df = df.loc[(df["y"] != '') & (~df["y"].isna())]

# Text vector change
tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
X1 = tfidfconverter.fit_transform(df["Interaction content"]).toarray()
X2 = tfidfconverter.transform(df["Ticket Summary"]).toarray()
X = np.concatenate((X1, X2), axis=1)

# Filter to labels
filtered_df = df.dropna(subset=['Type 2', 'Type 3', 'Type 4'])
filtered_df = filtered_df[filtered_df['Type 2'] != '']
filtered_df = filtered_df[filtered_df['Type 3'] != '']
filtered_df = filtered_df[filtered_df['Type 4'] != '']

y = filtered_df[['Type 2', 'Type 3', 'Type 4']]
X = X[filtered_df.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model selection
svm_model = MultiOutputClassifier(SVC(kernel='linear', probability=True, random_state=42))
svm_model.fit(X_train, y_train)

# Save the model and vectorizer
joblib.dump(tfidfconverter, 'svm_tfidf_vectorizer.pkl')
joblib.dump(svm_model, 'SVMModel.pkl')
