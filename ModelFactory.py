from models.modelClass.HDBCModel import HDBCModel
from models.modelClass.SVMModel import SVMModel

class ModelFactory():
    _models = {
        "HDBC": HDBCModel,
        "SVM": SVMModel
    }

    def get_model(model_name):
        model = ModelFactory._models.get(model_name.lower())
        if not model:
            raise ValueError(f"Model '{model_name}' is not supported " )        
        return model