import os
import subprocess

from ModelFactory import ModelFactory
from userSettings import userSettings
from decorator import log_function_call


def ensure_model_exists(model_name):
    """
    Ensures that the model files exist. If not, trains the model using the corresponding trainer.
    """
    model_paths = {
        "HGBC": "./exported_models/HGBC/HGBCModel.pkl",
        "SVM": "./exported_models/SVM/SVMModel.pkl",
        "KNN": "./exported_models/KNN/KNNModel.pkl",
        "NB": "./exported_models/NB/NaiveBayesModel.pkl",
        "CB": "./exported_models/CB/CatBoostModel.pkl",
    }

    trainer_scripts = {
        "HGBC": "models/modelTrainer/HGBCModelTrainer.py",
        "SVM": "models/modelTrainer/SVMModelTrainer.py",
        "KNN": "models/modelTrainer/NearestNeighborsTrainer.py",
        "NB": "models/modelTrainer/NaiveBayesTrainer.py",
        "CB": "models/modelTrainer/CatBoostTrainer.py",
    }

    model_path = model_paths.get(model_name)
    trainer_script = trainer_scripts.get(model_name)

    if not model_path or not trainer_script:
        raise ValueError(f"Invalid model name: {model_name}")

    if not os.path.exists(model_path):
        print(f"Model '{model_name}' not found. Training the model using {trainer_script}...")
        subprocess.run(["python", trainer_script], check=True)
        print(f"Model '{model_name}' trained and saved successfully.")


@log_function_call
def classify_email():
    """
    Performs email classification using the selected model(s) in the configuration.
    """
    subject = input("Please enter the subject line: ").strip()
    email = input("Please enter the email body text: ").strip()

    models = configuration.ml_models
    if "Run All Models" in models:
        models = ["HGBC", "SVM", "NB", "KNN"]

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

@log_function_call
def get_model_choice():
    """
    Allows the user to select one or more models and preprocessing options.
    Updates the global configuration accordingly.
    """
    print("Choose a model for classification. Separate your choices with commas if selecting multiple:")
    print("1. HGBC Model")
    print("2. SVM Model")
    print("3. Naive Bayes Model")
    print("4. Nearest Neighbors Model")
    print("5. CatBoost Model")
    print("5. Run All Models")
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
                selected_models.append("NB")
            elif choice.strip() == "4":
                selected_models.append("KNN")
            elif choice.strip() == "5":
                selected_models.append("CB")
            elif choice.strip() == "6":
                selected_models = ["HGBC", "SVM", "NB", "KNN"]
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

    for model in selected_models:
        ensure_model_exists(model)

    configuration.update_settings(selected_models, translate, noise_removal, verbose, explainable)
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
