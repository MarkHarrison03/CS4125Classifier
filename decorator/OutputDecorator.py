import functools
from tokenize import String

from deep_translator import GoogleTranslator
import re

class Verbose:
    @staticmethod
    def printVerbose(results):
        if not isinstance(results, dict):
             print("[VERBOSE] CatBoost does not support verbose.")
             return
        print("[VERBOSE] Verbose results:")

        for model_name, item in results.items():
            print(f"Model: {model_name}")
            if model_name == "CB":
                print("CatBoost does not support verbose.")
                return
            for key, value in item.items():
                if key == 'prediction':
                    print(f"  - {key}: {value.tolist()}")
                elif isinstance(value, (dict, list)):
                    print(f"  - {key}:")
                    Verbose.printVerbose({key: value})
                else:
                    print(f"  - {key}: {value}")

class OutputDecorator:
    def __init__(self, text_transformation: list):
        self.text_transformation = text_transformation

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper( *args, **kwargs):
            self.results = args[0]
            resolved_transformations = [
                transformation() if callable(transformation) else transformation
                for transformation in self.text_transformation
            ]

            if resolved_transformations[0]:
                try:
                    # if  isinstance(self.results, str):
                    #     print("[VERBOSE] CatBoost does not support verbose.")
                    #     print(self.results)
                    #     return func(self.results)

                      Verbose.printVerbose(self.results)
                      return func(self.results)
                except Exception as e:
                    print(f"Verbose printing failed: {e}")

            return func(self.results)

        return wrapper

