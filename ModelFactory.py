from models.modelClass.HGBCModel import HGBCModel
from models.modelClass.SVMModel import SVMModel
from models.modelClass.NearestNeighborModel import KNNModel
from models.modelClass.NaiveBayesModel import NBModel

class ModelFactory():
    _models = {
        "hgbc": HGBCModel,
        "svm": SVMModel,
        "knn": KNNModel,
        "nb": NBModel
    }

    def get_model(model_name):
        print(model_name)
        model = ModelFactory._models.get(model_name.lower())
        if not model:
            raise ValueError(f"Model '{model_name}' is not supported " )        
        return model