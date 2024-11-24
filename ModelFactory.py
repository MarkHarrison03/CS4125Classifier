import os
import importlib
from models.modelClass.ModelInterface import IModel


class ModelFactory:
    _models = {
        "hgbc": "models.modelClass.HGBCModel.HGBCModel",
        "svm": "models.modelClass.SVMModel.SVMModel",
        "knn": "models.modelClass.NearestNeighborModel.KNNModel",
        "nb": "models.modelClass.NaiveBayesModel.NBModel",
        "cb": "models.modelClass.CatBoostModel.CatBoostModel"
    }

    _model_paths = {
        "hgbc": "./exported_models/HGBC/HGBCModel.pkl",
        "svm": "./exported_models/SVM/SVMModel.pkl",
        "knn": "./exported_models/KNN/KNNModel.pkl",
        "nb": "./exported_models/NB/NaiveBayesModel.pkl",
        "cb": "./exported_models/CB/CatBoostModel.pkl"
    }

    @staticmethod
    def ensure_model_exists(model_name: str):
        """
        Ensure that the specified model file exists.
        """
        model_key = model_name.lower()
        model_path = ModelFactory._model_paths.get(model_key)

        if not model_path:
            raise ValueError(f"No configuration found for model '{model_name}'.")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file for '{model_name}' not found at '{model_path}'.")

    @staticmethod
    def get_model(model_name: str) -> IModel:
        """
        Get the model instance after ensuring the model file exists.
        """
        model_key = model_name.lower()
        ModelFactory.ensure_model_exists(model_key)

        model_path = ModelFactory._models.get(model_key)
        if not model_path:
            raise ValueError(f"Model '{model_name}' is not supported.")

        module_name, class_name = model_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        if not issubclass(model_class, IModel):
            raise TypeError(f"Model '{model_name}' does not implement the IModel interface.")

        return model_class()
