import os
import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from translate import Translator


class DataPreprocessor:
    def __init__(self, max_features=2000, target_language="en", data_folder="./data"):
        self.translator = Translator(to_lang=target_language)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, min_df=4, max_df=0.90)
        self.noise_patterns = [
            r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
            r"\d+",
            r"http\S+|www\S+",
            r"<.*?>",
            r"[^a-zA-Z\s]",
            r"\s+"
        ]
        self.data_folder = data_folder

    def preprocess_datasets(self, required_columns, translate=True):
        print("Loading datasets from folder...")
        all_dataframes = []
        for file_name in os.listdir(self.data_folder):
            if file_name.endswith(".csv"):
                file_path = os.path.join(self.data_folder, file_name)
                print(f"Processing file: {file_path}")
                try:
                    df = pd.read_csv(file_path)
                    # Ensure all required columns are present
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = None
                    all_dataframes.append(df[required_columns])
                except Exception as e:
                    print(f"Failed to load {file_name}. Error: {e}")

        if all_dataframes:
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            print(f"Combined dataset: {combined_df.shape[0]} rows, {combined_df.shape[1]} columns.")
            return combined_df
        else:
            print("No valid CSV files found.")
            return pd.DataFrame(columns=required_columns)

    def preprocess_text(self, text, translate=True):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        for pattern in self.noise_patterns:
            text = re.sub(pattern, " ", text)
        text = re.sub(r'\s+', ' ', text).strip()
        if translate:
            try:
                text = self.translator.translate(text)
            except Exception as e:
                print(f"Translation failed for text: {text}. Error: {e}")
        return text

    def preprocess_dataframe(self, df, content_col, summary_col, label_cols, translate=True):
        df[content_col] = df[content_col].apply(lambda x: self.preprocess_text(x, translate=translate))
        df[summary_col] = df[summary_col].apply(lambda x: self.preprocess_text(x, translate=translate))
        valid_indices = df.dropna(subset=label_cols).index
        df = df.loc[valid_indices]
        content_vectors = self.tfidf_vectorizer.fit_transform(df[content_col]).toarray()
        summary_vectors = self.tfidf_vectorizer.transform(df[summary_col]).toarray()
        features = np.concatenate((content_vectors, summary_vectors), axis=1)
        labels = df[label_cols]
        return features, labels

    def save_vectorizer(self, path):
        joblib.dump(self.tfidf_vectorizer, path)
