from command.command import Command
from utils.utils import classify_email, get_model_choice
from analytics.AnalyticsFacade import AnalyticsFacade
from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton


class ClassificationCommand(Command):
    def __init__(self, configuration):
        self.configuration = configuration

    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()


        if not self.configuration.ml_models:
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
        user_settings = UserSettingsSingleton.get_instance()


        get_model_choice()



class AnalyticsCommand(Command):
    def __init__(self):
        self.analytics = AnalyticsFacade()

    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()


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



        print("Exiting the application.")
        exit(0)
