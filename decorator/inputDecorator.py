import functools
from deep_translator import GoogleTranslator
import re

class TextTransformation:
    def translate(self, text):
        raise NotImplementedError
    def clean(self, text):
        raise NotImplementedError

class inputDecorator:
    def __init__(self, text_transformation: list, target_language="en"):
        self.text_transformation = text_transformation
        self.target_language = target_language

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper( *args, **kwargs):
            subject = input("Please enter the subject line: ").strip()
            email = input("Please enter the email body text: ").strip()

            # Dynamically resolve transformation settings
            resolved_transformations = [
                transformation() if callable(transformation) else transformation
                for transformation in self.text_transformation
            ]

            if resolved_transformations[0] == True:
                translator = GoogleTranslator(target=self.target_language)
                try:
                    subject = translator.translate(subject)
                    email = translator.translate(email)
                    print("Translated:", subject, email)
                except Exception as e:
                    print(f"Translation failed: {e}")

            if resolved_transformations[1] == True:
                noiseremover = NoiseRemoval()
                subject = noiseremover.clean(subject)
                email = noiseremover.clean(email)

            return func(subject, email, *args, **kwargs)

        return wrapper

class NoiseRemoval:
    @staticmethod
    def translate(text):
        return text
    @staticmethod
    def clean(text):
        noise_patterns = [
            r"\d{2}(:|.)\d{2}",
            r"(xxxxx@xxxx\.com)|(\*{5}\([a-z]+\))",
            r"dear ((customer)|(user))",
            r"dear",
            r"(hello)|(hallo)|(hi )|(hi there)",
            r"good morning",
            r"thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            r"thank you for contacting us",
            r"thank you for your availability",
            r"thank you for providing us this information",
            r"thank you for contacting",
            r"thank you for reaching us (back)?",
            r"thank you for patience",
            r"thank you for (your)? reply",
            r"thank you for (your)? response",
            r"thank you for (your)? cooperation",
            r"thank you for providing us with more information",
            r"thank you very kindly",
            r"thank you( very much)?",
            r"i would like to follow up on the case you raised on the date",
            r"i will do my very best to assist you",
            r"in order to give you the best solution",
            r"could you please clarify your request with following information:",
            r"in this matter",
            r"we hope you(( are)|('re)) doing ((fine)|(well))",
            r"i would like to follow up on the case you raised on",
            r"we apologize for the inconvenience",
            r"sent from my huawei (cell )?phone",
            r"original message",
            r"customer support team",
            r"(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            r"(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            r"canada, australia, new zealand and other countries",
            r"\d+",
            r"[^0-9a-zA-Z\s]+",
            r"(\s|^).(\s|$)"
        ]

        combined_pattern = "|".join(noise_patterns)
        cleaned_text = re.sub(combined_pattern, "", text.lower())
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        return cleaned_text
