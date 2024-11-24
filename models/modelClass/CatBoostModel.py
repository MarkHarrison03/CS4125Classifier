import numpy as np
import joblib
import os
from models.modelClass.ModelInterface import IModel


class CatBoostModel(IModel):
    def categorize(self, subject, email):
        print("WE IN HERE")
        tfidfconverter = joblib.load( './exported_models/CB/cb_tfidf_vectorizer.pkl')
        print("WE IN HERE5")
        print(tfidfconverter)

        classifier = joblib.load('./exported_models/CB/CatBoostModel.pkl')

        print("WE IN HERE6")

        print(classifier)

        print("WE IN HERE2")

        x1 = tfidfconverter.transform([email]).toarray()
        x2 = tfidfconverter.transform([subject]).toarray()
        X = np.concatenate((x1, x2), axis=1)
        print(f"Shape of X_new: {X.shape}")
        print("WE IN HERE3")

        y_pred = classifier.predict(X)
        print(f"Type of y_pred: {type(y_pred)}")


        # analytics = {
        #     "tfidf_vectorizer": tfidfconverter,
        #     "classifier_model": classifier,
        #     "x1_shape": x1.shape,
        #     "x2_shape": x2.shape,
        #     "X_shape": X.shape,
        #     "prediction": y_pred
        # }
        # print("Analytics Dictionary:", analytics)  # Should print the dictionary properly
        print(y_pred)
        return y_pred
