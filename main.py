from models.modelClass.HGBCModel import HGBCModel
from models.modelClass.SVMModel import SVMModel
from models.modelClass.NearestNeighborModel import KNNModel
from userSettings import userSettings
from ModelFactory import ModelFactory


def get_model_choice():
    """
    Allows the user to select one or more models and preprocessing options.
    Updates the global configuration accordingly.
    """
    print("\nChoose a model for classification. Separate your choices with commas if selecting multiple:")
    print("1. HGBC Model")
    print("2. SVM Model")
    print("3. Naive Bayes Model")
    print("4. Nearest Neighbors Model")
    print("5. Secret Other Model")
    print("6. Run All Models")
    choice_model = input("Enter your choice (e.g., 1,2): ").strip()

    selected_models = []
    if choice_model:
        choices = choice_model.split(",")
        for choice in choices:
            if choice.strip() == "1":
                selected_models.append("HGBC")
            elif choice.strip() == "2":
                selected_models.append("SVM")
            elif choice.strip() == "3":
                selected_models.append("nb")
            elif choice.strip() == "4":
                selected_models.append("knn")
            elif choice.strip() == "5":
                selected_models.append("Secret Other Model")
            elif choice.strip() == "6":
                selected_models = ["HGBC", "SVM", "nb", "knn", "Secret Other Model"]
                break

    if not selected_models:
        print("No models selected. Returning to main menu.")
        return

    print("\nChoose preprocessing settings:")
    print("1. Automatic text translation")
    print("2. Noise removal")
    choice_pre = input("Enter your choice (e.g., 1,2): ").strip()

    translate = False
    noise_removal = False
    if choice_pre:
        choices = choice_pre.split(",")
        for choice in choices:
            if choice.strip() == "1":
                translate = True
            elif choice.strip() == "2":
                noise_removal = True

    print("\nChoose postprocessing settings:")
    print("1. Verbose output")
    print("2. Explainable AI")
    choice_post = input("Enter your choice (e.g., 2): ").strip()

    verbose = False
    explainable = False
    if choice_post:
        choices = choice_post.split(",")
        for choice in choices:
            if choice.strip() == "1":
                verbose = True
            elif choice.strip() == "2":
                explainable = True

    configuration.update_settings(
        ml_model=selected_models,
        translate_text=translate,
        remove_noise=noise_removal,
        verbose=verbose,
        explainable=explainable,
    )
    print("\nConfiguration updated:")
    print(configuration)


def main_menu():
    """
    Displays the main menu and returns the user's choice.
    """
    print("\nWelcome to the Email Classifier.")
    print("1. Classification")
    print("2. Configuration")
    print("3. Analytics")
    print("4. Exit")
    return input("Choose an option (1/2/3/4): ").strip()


def classify_email():
    """
    Performs email classification using the selected model(s) in the configuration.
    """
    subject = input("Please enter the subject line: ").strip()
    email = input("Please enter the email body text: ").strip()

    models = configuration.ml_models
    if "Run All Models" in models:
        models = ["HGBC", "SVM", "nb", "knn", "Secret Other Model"]

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


configuration = userSettings()

while True:
    choice = main_menu()
    if choice == "1":
        if not configuration.ml_models:
            print("No models selected. Redirecting to Configuration...")
            get_model_choice()
            if not configuration.ml_models:
                print("Model selection is required to proceed. Returning to main menu.")
                continue
        classify_email()
    elif choice == "2":
        get_model_choice()
    elif choice == "3":
        print("Analytics functionality is not implemented yet. Stay tuned!")
    elif choice == "4":
        print("Exiting the program. Goodbye!")
        exit()
    else:
        print("Invalid choice. Please select a valid option.")
