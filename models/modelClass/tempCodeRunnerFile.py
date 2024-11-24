import numpy as np
import joblib


class CatBoostModel:
    def categorize(subject, email):
        
        print("HELLLOOOOO")
        tfidfconverter = joblib.load('./exported_models/CB/cb_tfidf_vectorizer.pkl')
        classifier = joblib.load('./exported_models/CB/CatBoostModel.pkl')
        print(f"Classifier attributes: {dir(classifier)}")
        num_features = len(tfidfconverter.get_feature_names_out())
        print(f"Vectorizer has {num_features} features.")
        subject_vec = tfidfconverter.transform([subject]).toarray()
        email_vec = tfidfconverter.transform([email]).toarray()
        X = np.concatenate((subject_vec, email_vec), axis=1)
        print(f"Shape of X (subject + email vectors): {X.shape}")
        
        # Check the expected number of features
        expected_features = classifier.n_features_in_
        print(f"Classifier expects {expected_features} features.")
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