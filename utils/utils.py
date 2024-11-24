import csv

import numpy as np

from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton
from ModelFactory import ModelFactory
from decorator.decorator import log_function_call
from decorator.inputDecorator import inputDecorator
from strategy.classificationStrategies import QuickStrategy, VerboseStrategy, NoiseRemovalStrategy, TranslateStrategy, HighPerformanceStrategy
import os

@inputDecorator([lambda: UserSettingsSingleton.get_instance().translate_text, 
                 lambda: UserSettingsSingleton.get_instance().remove_noise], 
                target_language="en")
@log_function_call
def classify_email(subject, email):
    configuration = UserSettingsSingleton.get_instance()

    models = configuration.ml_models
    if "Run All Models" in models:
        models = ["HGBC", "SVM", "NB", "KNN", "CB"]

    print(f"Final Inputs to Models: Subject: {subject}, Email: {email}")
    results = {}
    for model_name in models:
        try:
            classifier = ModelFactory.get_model(model_name)
            classification = classifier.categorize(subject, email)
            print("LOG AFTER", subject, email)
            results[model_name] = classification
        except ValueError as e:
            results[model_name] = f"Error: {e}"

    print("\nClassification Results:")
    for model_name, result in results.items():
        print(f"- {model_name} Model: {result}")
    save_classification_to_csv(subject, email, results, selected_models=models)
    return results, subject, email

def save_classification_to_csv(subject, email, results, selected_models, all_models=None, filename="classification_results.csv"):
    """
    Saves classification results to a CSV file with proper distribution into type-specific columns.
    Ensures no double quotes are added in the CSV output and replaces double quotes with single quotes.
    """
    if all_models is None:
        all_models = ["HGBC", "SVM", "NB", "KNN", "CB"]

    # Prepare headers dynamically for all models and types
    fieldnames = ["subject", "email"] + [f"{model}_Type{i}" for model in all_models for i in range(1, 5)]

    # Prepare a single dictionary for the row
    row = {"subject": subject, "email": email}

    for model_name in selected_models:
        classifications = results.get(model_name, [])  # Results for this model (list of lists)
        if isinstance(classifications, np.ndarray):
            classifications = classifications.tolist()  # Convert numpy array to a list

        # Check and flatten extra nesting for "CB" or other models
        while len(classifications) > 0 and isinstance(classifications[0], list):
            classifications = classifications[0]  # Unwrap one level of nesting

        for i in range(1, 5):  # Iterate over 4 types
            if i <= len(classifications):  # Ensure the type index exists in the results
                classification = classifications[i - 1]  # Get classification for this type
                # Convert to string for CSV storage
                if isinstance(classification, (list, np.ndarray)):
                    row[f"{model_name}_Type{i}"] = ",".join(map(str, classification)) if classification else ""
                else:
                    row[f"{model_name}_Type{i}"] = str(classification) if classification else ""
            else:
                # Fill empty types with an empty string
                row[f"{model_name}_Type{i}"] = ""

    # Write the row to the CSV
    try:
        file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            # Configure the CSV writer to avoid quoting
            writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_NONE, escapechar='\\')
            if not file_exists:  # If the file is new or empty, write the header
                writer.writeheader()
            writer.writerow(row)
        print(f"Classification result saved to {filename}.")
    except Exception as e:
        print(f"Error saving classification result to CSV: {e}")



@log_function_call
def get_model_choice():
    configuration = UserSettingsSingleton.get_instance()

    """
    Allows the user to select one or more models and preprocessing options.
    Updates the global configuration accordingly.
    """
    
    print("Would you like to use a preset strategy for classification, or customize your own?")
    print("1. Use a preset strategy")
    print("2. Customize your own strategy")
    choice = input("Enter your choice (1/2): ").strip()
    if choice == "1":
        use_preset_model_choice()
    elif choice == "2":
        customize_model_choice()
        
def use_preset_model_choice():
        print("Please choose your preset strategy: ")
        print("1. Quick Strategy: Single, lightweight model with minimal preprocessing")
        print("2. Verbose Strategy: All models are ran, with all preprocessing options enabled")
        print("3. Noise Reduction Strategy: Three models are ran, with only noise reduction enabled")
        print("4. Translation Strategy: Tgree models are ran, with only translation enabled")
        print("5. High Performance Strategy: Two lightweight models are ran, with only noise reduction enabled")
        
        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == "1":
            strategy = QuickStrategy(settings_manager=UserSettingsSingleton.get_instance())
        elif choice == "2":
            strategy = VerboseStrategy(settings_manager=UserSettingsSingleton.get_instance())
        elif choice == "3":
            strategy = NoiseRemovalStrategy(settings_manager=UserSettingsSingleton.get_instance())
        elif choice == "4":
            strategy = TranslateStrategy(settings_manager=UserSettingsSingleton.get_instance())
        elif choice == "5":
            strategy = HighPerformanceStrategy(settings_manager=UserSettingsSingleton.get_instance())
        else:
            print("Invalid choice! Please choose a valid strategy.")
            return  # Exit the function if invalid input is entered
        
        strategy.configure_context()
        print("Preset strategy applied successfully.")
        print(UserSettingsSingleton.get_instance())

def customize_model_choice():
    configuration = UserSettingsSingleton.get_instance()

    print("Choose a model for classification. Separate your choices with commas if selecting multiple:")
    print("1. HGBC Model")
    print("2. SVM Model")
    print("3. Naive Bayes Model")
    print("4. Nearest Neighbors Model")
    print("5. CatBoost Model")
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
                selected_models.append("NB")
            elif choice.strip() == "4":
                selected_models.append("KNN")
            elif choice.strip() == "5":
                selected_models.append("CB")
            elif choice.strip() == "6":
                selected_models = ["HGBC", "SVM", "NB", "KNN", "CB"]
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

    for model in selected_models:
        ensure_model_exists(model)

    configuration.update_settings(selected_models, translate, noise_removal)
    print("\nConfiguration updated:")
    print(configuration)


def ensure_model_exists(model_name):
    """
    Ensures that the model files exist. If not, trains the model using the corresponding trainer.
    """
    configuration = UserSettingsSingleton.get_instance()

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

