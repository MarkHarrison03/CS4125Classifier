import sys
import os
import tkinter as tk
from tkinter import messagebox, Toplevel, StringVar, Text
from tkinter.ttk import Combobox
from command.classificationCommand import ClassificationCommand, ConfigureCommand, AnalyticsCommand
from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton
from utils.ensuremodelexists import ensure_models_exist
from utils.utils import log_function_call
from strategy.classificationStrategies import QuickStrategy, VerboseStrategy, NoiseRemovalStrategy, TranslateStrategy, HighPerformanceStrategy
ensure_models_exist()
class EmailClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Email Classifier")
        self.root.geometry("400x300")
        
        self.configuration = UserSettingsSingleton.get_instance()

        self.classify_command = ClassificationCommand(self.configuration, root)
        self.analytics_command = AnalyticsCommand(root)

        title = tk.Label(self.root, text="Email Classifier", font=("Helvetica", 16), pady=20)
        title.pack()

        classification_button = tk.Button(self.root, text="Classification", command=self.open_classification_window, width=20)
        classification_button.pack(pady=5)

        configuration_button = tk.Button(self.root, text="Configuration", command=self.open_configuration_window, width=20)
        configuration_button.pack(pady=5)

        analytics_button = tk.Button(self.root, text="Analytics", command=self.open_analytics_window, width=20)
        analytics_button.pack(pady=5)

        strategies_button = tk.Button(self.root, text="Strategies", command=self.open_strategies_window, width=20)
        strategies_button.pack(pady=5)

        exit_button = tk.Button(self.root, text="Exit", command=self.exit_app, width=20)
        exit_button.pack(pady=5)

    @log_function_call
    def open_classification_window(self):
        classification_window = Toplevel(self.root)
        classification_window.title("Classify Email")
        classification_window.geometry("400x400")

        tk.Label(classification_window, text="Enter Email Subject:", font=("Helvetica", 12)).pack(pady=10)
        subject_input = StringVar()
        tk.Entry(classification_window, textvariable=subject_input, width=40).pack(pady=5)

        tk.Label(classification_window, text="Enter Email Body:", font=("Helvetica", 12)).pack(pady=10)
        body_input = Text(classification_window, height=10, width=40)
        body_input.pack(pady=5)

        def classify_email_action():
            subject = subject_input.get()
            body = body_input.get("1.0", "end-1c")
            self.classify_command.execute(subject, body)

        tk.Button(classification_window, text="Classify", command=classify_email_action, width=20).pack(pady=10)

    @log_function_call
    def open_configuration_window(self):
        configure_command = ConfigureCommand(self.configuration, self.root)
        configure_command.execute()

    @log_function_call
    def open_analytics_window(self):
        analytics_command = AnalyticsCommand(self.root)
        analytics_command.execute()

    @log_function_call
    def open_strategies_window(self):
        """Open the strategies selection window."""
        strategies_window = Toplevel(self.root)
        strategies_window.title("Select Strategy")
        strategies_window.geometry("400x300")

        tk.Label(strategies_window, text="Select Strategy", font=("Helvetica", 16), pady=10).pack()

        strategy_names = [
            "Quick Strategy",
            "Noise Removal Strategy",
            "Translate Strategy",
            "High Performance Strategy"
        ]

        strategy_map = {
            "Quick Strategy": QuickStrategy(self.configuration),
            "Noise Removal Strategy": NoiseRemovalStrategy(self.configuration),
            "Translate Strategy": TranslateStrategy(self.configuration),
            "High Performance Strategy": HighPerformanceStrategy(self.configuration),
        }

        strategy_var = StringVar(value="Select a Strategy")
        strategy_combobox = Combobox(
            strategies_window,
            textvariable=strategy_var,
            values=strategy_names,
            state="readonly"
        )
        strategy_combobox.pack(pady=10)

        def apply_strategy():
            strategy_name = strategy_var.get()
            if strategy_name in strategy_map:
                strategy = strategy_map[strategy_name]
                strategy.configure_context()
                messagebox.showinfo("Strategy Applied", f"{strategy_name} applied successfully!")
                strategies_window.destroy()
                self.open_classification_window()
            else:
                messagebox.showerror("Error", "Please select a valid strategy.")

        tk.Button(strategies_window, text="Apply Strategy", command=apply_strategy).pack(pady=10)
        tk.Button(strategies_window, text="Close", command=strategies_window.destroy).pack(pady=10)

    def exit_app(self):
        """Exits the application."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.root.quit()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmailClassifierApp(root)
    root.mainloop()
