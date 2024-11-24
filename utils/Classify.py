from decorator.inputDecorator import inputDecorator

from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton
from ModelFactory import ModelFactory


@inputDecorator([lambda: UserSettingsSingleton.get_instance().translate_text, lambda: UserSettingsSingleton.get_instance().remove_noise],
        target_language="en")
def classify_email(subject, email):
    configuration = UserSettingsSingleton.get_instance()
    print("config" , UserSettingsSingleton.get_instance().remove_noise)
    """
    Performs email classification using the selected model(s) in the configuration.
    """
    #subject = input("Please enter the subject line: ").strip()
    #email = input("Please enter the email body text: ").strip()

    models = configuration.ml_models
    if "Run All Models" in models:
        models = ["HGBC", "SVM", "NB", "KNN", "CB"]
    print(subject, email)
    results = {}
    for model_name in models:
        try:
            classifier = ModelFactory.get_model(model_name)
            classification = classifier.categorize(subject, email)
            results[model_name] = classification
        except ValueError as e:
            results[model_name] = f"Error: {e}"

    print("\nClassification Results:")
    for model_name, result in results.items():
        print(f"- {model_name} Model: {result}")
   # save_classification_to_csv(subject, email, results, selected_models=models)




