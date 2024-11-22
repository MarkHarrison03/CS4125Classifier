import pandas as pd
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer 
from io import StringIO

import numpy as np
class HGBCModel:
    def categorize(subject, email):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))  
        tfidfconverter = joblib.load(os.path.join(root_dir, 'exported_models/HGBC/tfidf_vectorizer.pkl'))
        classifier = joblib.load(os.path.join(root_dir,'exported_models/HGBC/HGBCModel.pkl'))

        x1 = tfidfconverter.transform([email]).toarray()
        x2 = tfidfconverter.transform([subject]).toarray()
        
        X = np.concatenate((x1, x2), axis=1)

        print(f"Shape of X_new: {X.shape}")
        y_pred = classifier.predict(X)
        
        return y_pred