import numpy as np
import joblib
import os
from models.modelClass.ModelInterface import IModel
class HGBCModel(IModel):
    def categorize(self, subject, email):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))

        tfidfconverter = joblib.load(os.path.join(root_dir, 'exported_models/HGBC/tfidf_vectorizer.pkl'))
        classifier = joblib.load(os.path.join(root_dir, 'exported_models/HGBC/HGBCModel.pkl'))

        # Transform the email and subject into TF-IDF feature vectors
        x1 = tfidfconverter.transform([email]).toarray()
        x2 = tfidfconverter.transform([subject]).toarray()
        print(f"Shape of x1 (email vector): {x1.shape}")
        print(f"Shape of x2 (subject vector): {x2.shape}")

        # Check if the dimensions are compatible
        if x1.shape[0] != x2.shape[0]:
            raise ValueError(f"Incompatible dimensions: x1 has {x1.shape[0]} rows, x2 has {x2.shape[0]} rows")

        # Concatenate the two vectors into one feature array
        X = np.concatenate((x1, x2), axis=1)
        print(f"Shape of concatenated features (X): {X.shape}")

        print(f"Shape of X_new: {X.shape}")
        y_pred = classifier.predict(X)

        analytics = {
            "root_dir": root_dir,
            "tfidf_vectorizer": tfidfconverter,
            "classifier_model": classifier,
            "x1_shape": x1.shape,
            "x2_shape": x2.shape,
            "X_shape": X.shape,
            "prediction": y_pred
        }

        return analytics

