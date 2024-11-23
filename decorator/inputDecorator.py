import functools
from googletrans import Translator
from userSettings import userSettings

 

class inputDecorator:
    def __init__(self, target_language="en"):

        self.translator = Translator()
        self.target_language = target_language
        
        
    def translate_text(self, text):
        """Uses translation API to translate the text."""
        try:
            
            translation = self.translator.translate(text, dest="en")
            print(translation)
            return translation
        except Exception as e:
            print(f"Error translating text: {e}")
            return text 

    def __call__(self, func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            user_settings = userSettings()   
            self.translate = user_settings.translate_text
            print(user_settings)
            print("Aaa!")
            print(args, kwargs)
            print(self.translate)
            if self.translate:
                subject = input("Please enter the subject line: ").strip()
                email = input("Please enter the email body text: ").strip()

                translated_subject = self.translate_text(subject)
                translated_body = self.translate_text(email)
                
                return func(translated_subject.text, translated_body.text, *args, **kwargs)
            return func(*args, **kwargs)
        
        return wrapper
