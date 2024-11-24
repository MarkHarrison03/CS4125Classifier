import numpy as np
import joblib
import os
from models.modelClass.ModelInterface import IModel


class CatBoostModel(IModel):
    def categorize(self, subject, email):
        tfidfconverter = joblib.load( './exported_models/CB/cb_tfidf_vectorizer.pkl')
        print("HALLO1")

        classifier = joblib.load('./exported_models/CB/CatBoostModel.pkl')
        print("HALLO2")
        x1 = tfidfconverter.transform([email]).toarray()
        x2 = tfidfconverter.transform([subject]).toarray()
        X = np.concatenate((x1, x2), axis=1)

        y_pred = classifier.predict(X)
        results = {"prediction": y_pred}
        print("catboost results", y_pred)
        return y_pred
