import functools
from deep_translator import GoogleTranslator
import re

class Verbose:
    @staticmethod
    def printVerbose(results):
        print("[VERBOSE] Verbose results:")
        for model_name, item in results.items():
            print(f"Model: {model_name}")
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
                    Verbose.printVerbose(self.results)

                except Exception as e:
                    print(f"Translation failed: {e}")

            return func(self.results)

        return wrapper

