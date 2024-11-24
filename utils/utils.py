from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton
from ModelFactory import ModelFactory
from strategy.classificationStrategies import QuickStrategy, VerboseStrategy, NoiseRemovalStrategy, TranslateStrategy, HighPerformanceStrategy
import os


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
        print("1. Quick Strategy: Single, lightweight model with minimal pre and post processing")
        print("2. Verbose Strategy: All models are ran, with all pre and post processing options enabled")
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

    print("\nChoose postprocessing settings:")
    print("1. Verbose output")
    choice_post = input("Enter your choice (e.g. 1): ").strip()

    verbose = False
    explainable = False
    if choice_post:
        choices = choice_post.split(",")
        for choice in choices:
            if choice.strip() == "1":
                verbose = True


    for model in selected_models:
        ensure_model_exists(model)

    configuration.update_settings(selected_models, translate, noise_removal, verbose, explainable)
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

