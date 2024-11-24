import os
import sys
import subprocess
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from userSettings import userSettings
from decorator.decorator import log_function_call
from decorator.inputDecorator import inputDecorator
from command.classificationCommand import ClassificationCommand, ConfigureCommand, AnalyticsCommand, ExitCommand
from invoker.MenuInvoker import MenuInvoker

def main_menu():
    """
    Displays the main menu and returns the user's choice.
    """
    print("\nWelcome to the Email Classifier.")
    print("1. Classification")
    print("2. Configuration")
    print("3. Analytics")
    print("4. Exit")
    return input("Choose an option (1/2/3/4): ").strip()


configuration = userSettings()
classify_command = ClassificationCommand(configuration)
configure_command = ConfigureCommand(configuration)
analytics_command = AnalyticsCommand()
exit_command = ExitCommand()

menu_invoker = MenuInvoker()
menu_invoker.register_command("1", classify_command)
menu_invoker.register_command("2", configure_command)
menu_invoker.register_command("3", analytics_command)
menu_invoker.register_command("4", exit_command)

while True:
    choice = main_menu()
    menu_invoker.execute_command(choice)
    