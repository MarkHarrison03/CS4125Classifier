from command.command import Command
from utils.utils import get_model_choice
from utils.Classify import classify_email
from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton


class ClassificationCommand(Command):
    def __init__(self, configuration):
        self.configuration = configuration

    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.verbose:
            print(f"[VERBOSE] Executing ClassificationCommand with configuration: {self.configuration}")

        if not self.configuration.ml_models:
            print("No models selected. Returning to main menu.")
            if user_settings.explainable:
                print("[EXPLAINABLE] No models are configured. The system is redirecting to configuration.")

            configure_command = ConfigureCommand(self.configuration)
            configure_command.execute()

            if not self.configuration.ml_models:
                print("No models selected. Returning to main menu.")
                if user_settings.verbose:
                    print("[VERBOSE] Exiting ClassificationCommand as no models were selected after configuration.")
                return

        if user_settings.verbose:
            print(f"[VERBOSE] Classifying email using selected models: {self.configuration.ml_models}")
        if user_settings.explainable:
            print("[EXPLAINABLE] The email classification will use the configured machine learning models.")

        classify_email()


class ConfigureCommand(Command):
    def __init__(self, configuration):
        self.configuration = configuration

    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.verbose:
            print(f"[VERBOSE] Executing ConfigureCommand with configuration: {self.configuration}")

        if user_settings.explainable:
            print("[EXPLAINABLE] The system is prompting the user to select machine learning models.")

        get_model_choice()

        if user_settings.verbose:
            print("[VERBOSE] Updated configuration after model selection.")


class AnalyticsCommand(Command):
    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.verbose:
            print("[VERBOSE] Executing AnalyticsCommand.")
        if user_settings.explainable:
            print("[EXPLAINABLE] This command provides analytics insights.")

        print("Analytics command executed.")


class ExitCommand(Command):
    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.verbose:
            print("[VERBOSE] Executing ExitCommand.")
        if user_settings.explainable:
            print("[EXPLAINABLE] Exiting the application after completing all tasks.")

        print("Exiting the application.")
        exit(0)
