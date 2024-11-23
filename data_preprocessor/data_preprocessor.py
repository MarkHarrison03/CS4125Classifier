import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from translate import Translator

class DataPreprocessor:
    def __init__(self, max_features=2000, target_language="en"):
        self.translator = Translator(to_lang=target_language)  # Initialize the translator
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, min_df=4, max_df=0.90)
        self.noise_patterns = [
            r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
            r"\d+",
            r"http\S+|www\S+",
            r"<.*?>",
            r"[^a-zA-Z\s]",
            r"\s+"
        ]

    def preprocess_text(self, text, translate=True):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()

        # Apply regex-based cleaning
        for pattern in self.noise_patterns:
            text = re.sub(pattern, " ", text)
        text = re.sub(r'\s+', ' ', text).strip()

        # Translate to the target language if specified
        if translate:
            try:
                text = self.translator.translate(text)
            except Exception as e:
                print(f"Translation failed for text: {text}. Error: {e}")
                # Fallback to original cleaned text if translation fails
        return text

    def preprocess_dataframe(self, df, content_col, summary_col, label_cols, translate=True):
        # Clean and translate text columns
        df[content_col] = df[content_col].apply(lambda x: self.preprocess_text(x, translate=translate))
        df[summary_col] = df[summary_col].apply(lambda x: self.preprocess_text(x, translate=translate))

        # Remove rows with missing labels
        valid_indices = df.dropna(subset=label_cols).index
        df = df.loc[valid_indices]

        # Vectorize processed content and summary
        content_vectors = self.tfidf_vectorizer.fit_transform(df[content_col]).toarray()
        summary_vectors = self.tfidf_vectorizer.transform(df[summary_col]).toarray()

        # Combine content and summary vectors
        features = np.concatenate((content_vectors, summary_vectors), axis=1)
        labels = df[label_cols]

        return features, labels

    def prepare_targets(self, df, target_columns, main_target):
        # Rename target columns
        for i, col in enumerate(target_columns, start=1):
            df[f"y{i}"] = df[col]
        # Set the main target column
        df["y"] = df[main_target]
        # Remove rows with empty or NaN values in the main target column
        df = df.loc[df["y"].notna() & (df["y"] != '')]
        return df

    def save_vectorizer(self, path):
        joblib.dump(self.tfidf_vectorizer, path)
