import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer 
from io import StringIO

import numpy as np
class HDBCModel:
    def categorize(subject, email):
        tfidfconverter = joblib.load('tfidf_vectorizer.pkl')
        classifier = joblib.load('HDBCModel.pkl')

        x1 = tfidfconverter.transform([email]).toarray()
        x2 = tfidfconverter.transform([subject]).toarray()
        
        X = np.concatenate((x1, x2), axis=1)

        print(f"Shape of X_new: {X.shape}")
        y_pred = classifier.predict(X)
        
        return y_pred