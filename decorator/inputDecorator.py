import functools
from googletrans import Translator
from userSettings import userSettings
import re
 

class inputDecorator:
    def __init__(self, target_language="en"):

        self.translator = Translator()
        self.target_language = target_language
        

    def remove_noise(self, text):
            noise_1 = [
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
                r"[^0-9a-zA-Z]+",
                r"(\s|^).(\s|$)"
            ]
            
            combined_pattern = "|".join(noise_1)
            cleaned_text = re.sub(combined_pattern, "", text.lower())
            cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            print( "cleaned text", cleaned_text)

            return cleaned_text
        
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
            subject = input("Please enter the subject line: ").strip()
            email = input("Please enter the email body text: ").strip()
            
            user_settings = userSettings() 
            print(user_settings)  
            self.translate = user_settings.translate_text
            self.removeNoise = user_settings.remove_noise
            print(self.translate, self.removeNoise)

            
            if self.translate == True:
                print("translated!")
                subject = self.translate_text(subject).text
                email = self.translate_text(email).text
                
            if self.removeNoise == True:
                subject = self.remove_noise(subject)
                email = self.remove_noise(email)
                print("Noise removed")
                print(subject, email)        
                return func(subject, email, *args, **kwargs)
       

            return func(subject, email, *args, **kwargs)

        
        return wrapper
