from command.command import Command
from utils.utils import classify_email, get_model_choice

class ClassificationCommand(Command):
    def __init__(self, configuration):
        self.configuration = configuration
        
    def execute(self):
        if not self.configuration.ml_models :
            print("No models selected. Returning to main menu.")
            configure_command = ConfigureCommand(self.configuration)
            configure_command.execute()
            if not self.configuration.ml_models:
                print("No models selected. Returning to main menu.")
                return
        classify_email()
        
class ConfigureCommand(Command):
    def __init__(self, configuration):
        self.configuration = configuration
    def execute(self):
        get_model_choice()
        
class AnalyticsCommand(Command):
    def execute(self):
        print("Analytics command executed.")
        
class ExitCommand(Command):
    def execute(self):
        print("Exiting the application.")
        exit(0)