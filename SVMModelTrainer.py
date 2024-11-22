import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from googletrans import Translator
import joblib
import numpy as np

translator = Translator()

def translate_text(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation failed for text: {text}. Error: {e}")
        return text

df = pd.read_csv("AppGallery.csv")

print("Translating content...")
df['Interaction content'] = df['Interaction content'].apply(translate_text)
df['Ticket Summary'] = df['Ticket Summary'].apply(translate_text)
print("Translation completed.")

df['Interaction content'] = df['Interaction content'].astype(str)
df['Ticket Summary'] = df['Ticket Summary'].astype(str)

df["y"] = df["Type 2"]
labels = df[['Type 2', 'Type 3', 'Type 4']]

df = df.loc[(df["y"] != '') & (~df["y"].isna())]

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
X1 = tfidfconverter.fit_transform(df["Interaction content"]).toarray()
X2 = tfidfconverter.transform(df["Ticket Summary"]).toarray()
X = np.concatenate((X1, X2), axis=1)

filtered_df = df.dropna(subset=['Type 2', 'Type 3', 'Type 4'])
filtered_df = filtered_df[filtered_df['Type 2'] != '']
filtered_df = filtered_df[filtered_df['Type 3'] != '']
filtered_df = filtered_df[filtered_df['Type 4'] != '']

y = filtered_df[['Type 2', 'Type 3', 'Type 4']]
X = X[filtered_df.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

svm_model = MultiOutputClassifier(SVC(kernel='linear', probability=True, random_state=42))
svm_model.fit(X_train, y_train)

joblib.dump(tfidfconverter, 'svm_tfidf_vectorizer.pkl')
joblib.dump(svm_model, 'SVMModel.pkl')

y_pred = svm_model.predict(X_test)

for i, label in enumerate(y.columns):
    print(f"\nClassification Report for {label}:")
    print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    print(f"Confusion Matrix for {label}:")
    print(confusion_matrix(y_test.iloc[:, i], y_pred[:, i]))

accuracies = []
for i, label in enumerate(y.columns):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracies.append(accuracy)
    print(f"Accuracy for {label}: {accuracy:.2f}")

average_accuracy = sum(accuracies) / len(accuracies)
print(f"\nAverage Accuracy across all labels: {average_accuracy:.2f}")

