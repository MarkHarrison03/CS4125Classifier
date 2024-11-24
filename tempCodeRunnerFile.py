from decorator.decorator import log_function_call
from decorator.inputDecorator import inputDecorator


@inputDecorator(target_language="en")
@log_function_call
def classify_email(subject, email):
    """
    Performs email classification using the selected model(s) in the configuration.
    """
    #subject = input("Please enter the subject line: ").strip()
    #email = input("Please enter the email body text: ").strip()

    models = configuration.ml_models
    if "Run All Models" in models:
        models = ["HGBC", "SVM", "NB", "KNN"]
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
