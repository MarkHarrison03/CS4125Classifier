import numpy as np
import joblib


class CatBoostModel:
    def categorize(subject, email):
        
        tfidfconverter = joblib.load('./exported_models/CB/cb_tfidf_vectorizer.pkl')
        classifier = joblib.load('./exported_models/CB/CatBoostModel.pkl')
        num_features = len(tfidfconverter.get_feature_names_out())
        subject_vec = tfidfconverter.transform([subject]).toarray()
        email_vec = tfidfconverter.transform([email]).toarray()
        X = np.concatenate((subject_vec, email_vec), axis=1)

        # Check the expected number of features
        expected_features = classifier.n_features_in_
        expected_features = classifier.n_features_in_
        actual_features = X.shape[1]
        if actual_features != expected_features:
            raise ValueError(
                f"Input has {actual_features} features, but the model expects {expected_features} features. "
                "Ensure the same TF-IDF vectorizer is used for training and prediction."
            )

        y_pred = classifier.predict(X)

        return y_pred

    categorize("subject", "email")