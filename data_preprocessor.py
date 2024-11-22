import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class DataPreprocessor:
    def __init__(self, max_features=102):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
        self.noise_patterns = [
            r"(from :)|(subject :)|(sent :)|(r\s*:)|(re\s*:)",
            r"(january|february|march|april|may|june|july|august|september|october|november|december)",
            r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            r"\d{2}(:|.)\d{2}",
            r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
            r"dear ((customer)|(user))",
            r"dear",
            r"(hello)|(hallo)|(hi )|(hi there)",
            r"good morning",
            r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            r"thank you for contacting us",
            r"thank you for your availability",
            r"thank you for providing us this information",
            r"thank you for contacting",
            r"thank you for reaching us (back)?",
            r"thank you for patience",
            r"thank you for (your)? reply",
            r"thank you for (your)? response",
            r"thank you for (your)? cooperation",
            r"thank you for providing us with more information",
            r"thank you very kindly",
            r"thank you( very much)?",
            r"i would like to follow up on the case you raised on the date",
            r"i will do my very best to assist you",
            r"in order to give you the best solution",
            r"could you please clarify your request with following information:",
            r"in this matter",
            r"we hope you(( are)|('re)) doing ((fine)|(well))",
            r"i would like to follow up on the case you raised on",
            r"we apologize for the inconvenience",
            r"sent from my huawei (cell )?phone",
            r"original message",
            r"customer support team",
            r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            r"canada, australia, new zealand and other countries",
            r"\d+",
            r"[^0-9a-zA-Z]+",
            r"(\s|^).(\s|$)"
        ]

    def preprocess_text(self, text):
        if not isinstance(text, str):  # Check if the value is not a string
            text = str(text)  # Convert non-string values to a string (or use an empty string: `text = ""`)
        # Lowercase the text
        text = text.lower()
        # Remove noise patterns
        for pattern in self.noise_patterns:
            text = re.sub(pattern, " ", text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def preprocess_dataframe(self, df, content_col, summary_col, label_cols):
        # Preprocess text columns
        df["processed_content"] = df[content_col].apply(self.preprocess_text)
        df["processed_summary"] = df[summary_col].apply(self.preprocess_text)

        # Remove rows with missing labels
        valid_indices = df.dropna(subset=label_cols).index
        df = df.loc[valid_indices]

        # Vectorize processed content and summary
        content_vectors = self.tfidf_vectorizer.fit_transform(df["processed_content"]).toarray()
        summary_vectors = self.tfidf_vectorizer.fit_transform(df["processed_summary"]).toarray()

        # Combine content and summary vectors
        features = np.concatenate((content_vectors, summary_vectors), axis=1)
        labels = df[label_cols]

        return features, labels

    def save_vectorizer(self, path):
        import joblib
        joblib.dump(self.tfidf_vectorizer, path)

    def load_vectorizer(self, path):
        import joblib
        self.tfidf_vectorizer = joblib.load(path)
