import os
import importlib
from models.modelClass.HGBCModel import HGBCModel
from models.modelClass.SVMModel import SVMModel
from models.modelClass.NearestNeighborModel import KNNModel
from models.modelClass.NaiveBayesModel import NBModel
from models.modelClass.CatBoostModel import CatBoostModel


class ModelFactory:
    _models = {
        "hgbc": HGBCModel,
        "svm": SVMModel,
        "knn": KNNModel,
        "nb": NBModel,
        "cb" : CatBoostModel
    }

    _model_paths = {
        "hgbc": ("./exported_models/HGBC/HGBCModel.pkl", "models.modelTrainer.HGBCModelTrainer"),
        "svm": ("./exported_models/SVM/SVMModel.pkl", "models.modelTrainer.SVMModelTrainer"),
        "knn": ("./exported_models/KNN/KNNModel.pkl", "models.modelTrainer.NearestNeighborsTrainer"),
        "nb": ("./exported_models/NB/NBModel.pkl", "models.modelTrainer.NaiveBayesTrainer"),
        "cb": ("./exported_models/CB/CatBoostModel.pkl", "models.modelTrainer.CatBoostTrainer")
    }

    @staticmethod
    def ensure_model_exists(model_name):
        """
        Ensure that the specified model file exists. If not, train the model using its respective trainer.
        """
        model_key = model_name.lower()
        model_path, trainer_module = ModelFactory._model_paths.get(model_key, (None, None))
        
        if not model_path or not trainer_module:
            raise ValueError(f"No configuration found for model '{model_name}'.")
        
        if not os.path.exists(model_path):
            print(f"Model file not found for '{model_name}'. Training the model...")
            trainer = importlib.import_module(trainer_module)
            if hasattr(trainer, "main"):
                trainer.main()
            else:
                raise ValueError(f"Trainer '{trainer_module}' does not have a 'main' function.")

    @staticmethod
    def get_model(model_name):
        """
        Get the model instance after ensuring the model file exists.
        """
        model_key = model_name.lower()
        ModelFactory.ensure_model_exists(model_key)

        model = ModelFactory._models.get(model_key)
        if not model:
            raise ValueError(f"Model '{model_name}' is not supported.")
        return model
