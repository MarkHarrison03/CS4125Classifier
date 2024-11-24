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

