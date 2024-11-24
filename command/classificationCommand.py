from command.command import Command
from utils.utils import classify_email, get_model_choice
from analytics.AnalyticsFacade import AnalyticsFacade
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
    def __init__(self):
        self.analytics = AnalyticsFacade()

    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.verbose:
            print("[VERBOSE] Executing AnalyticsCommand.")
        if user_settings.explainable:
            print("[EXPLAINABLE] This command provides analytics insights.")

        print("\n=== Analytics Menu ===")
        while True:
            choice = self.analytics.display_menu()
            if choice == "1":  # Compute Statistics
                model_names = self.analytics.get_model_choices()
                stats = self.analytics.compute_model_statistics(model_names)
                if stats:  # Ensure stats is not None
                    for type_label, percentages in stats.items():
                        print(f"\n{type_label}:")
                        for category, percentage in percentages.items():
                            print(f"  {category}: {percentage:.2f}%")
                else:
                    print("No results available to compute statistics.")

            elif choice == "2":  # Visualize Results
                model_names = self.analytics.get_model_choices()
                self.analytics.generate_visualization(model_names)

            elif choice == "3":  # Display Results
                model_names = self.analytics.get_model_choices()
                self.analytics.display_results(model_names)

            elif choice == "4":  # Exit
                print("Exiting Analytics Menu.")
                break

            else:
                print("Invalid choice. Please try again.")


class ExitCommand(Command):
    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.verbose:
            print("[VERBOSE] Executing ExitCommand.")
        if user_settings.explainable:
            print("[EXPLAINABLE] Exiting the application after completing all tasks.")

        print("Exiting the application.")
        exit(0)
