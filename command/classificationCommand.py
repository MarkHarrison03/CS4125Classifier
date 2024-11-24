from command.command import Command
from utils.utils import classify_email, get_model_choice
from analytics.AnalyticsFacade import AnalyticsFacade
from tkinter import messagebox, Toplevel, Label, ttk, Checkbutton, IntVar, Button
from tkinter.simpledialog import askstring
from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton
from tkinter.messagebox import showinfo

class ClassificationCommand(Command):
    def __init__(self, configuration, root):
        self.configuration = configuration
        self.root = root
        self.results_window = None
        self.tree = None

    def execute(self, subject, email):
        user_settings = UserSettingsSingleton.get_instance()

        if not self.configuration.ml_models:
            messagebox.showwarning("Configuration Missing", "No models selected. Redirecting to Configuration...")
            configure_command = ConfigureCommand(self.configuration, self.root)
            configure_command.execute()

            if not self.configuration.ml_models:
                messagebox.showerror("Error", "No models selected. Returning to main menu.")
                return

        try:
            results, finalSubject, finalEmail = classify_email(subject, email)
            self.show_results_window(finalSubject, finalEmail, results)
        except Exception as e:
            messagebox.showerror("Classification Error", f"An error occurred: {e}")

    def show_results_window(self, transformed_subject, transformed_email, results):
        """Display or update classification results in a window."""
        if self.results_window and self.results_window.winfo_exists():
            for model_name, result in results.items():
                self.tree.insert("", "end", values=(transformed_subject, transformed_email, model_name, result))
            return

        self.results_window = Toplevel(self.root)
        self.results_window.title("Classification Results")
        self.results_window.geometry("800x400")

        Label(self.results_window, text="Classification Results", font=("Helvetica", 16), pady=10).pack()

        self.tree = ttk.Treeview(
            self.results_window, columns=("Transformed Input", "Email", "Model", "Output"), show="headings"
        )
        self.tree.heading("Transformed Input", text="Transformed Input")
        self.tree.heading("Email", text="Email")
        self.tree.heading("Model", text="Model")
        self.tree.heading("Output", text="Output")

        self.tree.column("Transformed Input", width=200, anchor="center")
        self.tree.column("Email", width=200, anchor="center")
        self.tree.column("Model", width=150, anchor="center")
        self.tree.column("Output", width=250, anchor="center")

        for model_name, result in results.items():
            self.tree.insert(
                "", "end", values=(transformed_subject, transformed_email, model_name, ", ".join(result.flatten()))
            )

        self.tree.pack(expand=True, fill="both", padx=10, pady=10)

        ttk.Button(self.results_window, text="Close", command=self.results_window.destroy).pack(pady=10)

class ConfigureCommand:
    def __init__(self, configuration, root):
        self.configuration = configuration
        self.root = root

    def execute(self):
        """Open a configuration window for model and preprocessing settings."""
        config_window = Toplevel(self.root)
        config_window.title("Configuration")
        config_window.geometry("400x400")

        Label(config_window, text="Model Selection", font=("Helvetica", 12)).pack(pady=10)

        model_options = ["HGBC", "SVM", "NB", "KNN", "CB"]
        model_vars = {model: IntVar(value=model in self.configuration.ml_models) for model in model_options}

        for model in model_options:
            Checkbutton(config_window, text=model, variable=model_vars[model]).pack(anchor="w", padx=20)

        Label(config_window, text="Preprocessing Options", font=("Helvetica", 12)).pack(pady=10)
        translate_var = IntVar(value=self.configuration.translate_text)
        noise_var = IntVar(value=self.configuration.remove_noise)

        Checkbutton(config_window, text="Translate Text", variable=translate_var).pack(anchor="w", padx=20)
        Checkbutton(config_window, text="Remove Noise", variable=noise_var).pack(anchor="w", padx=20)

        def save_settings():
            selected_models = [model for model, var in model_vars.items() if var.get()]
            self.configuration.update_settings(
                ml_model=selected_models,
                translate_text=bool(translate_var.get()),
                remove_noise=bool(noise_var.get())
            )
            showinfo("Configuration Saved", f"Settings updated:\n\n{self.configuration}")
            config_window.destroy()

        Button(config_window, text="Save", command=save_settings).pack(pady=20)

        Button(config_window, text="Cancel", command=config_window.destroy).pack(pady=10)


class ExitCommand(Command):
    def execute(self):
        user_settings = UserSettingsSingleton.get_instance()

        if user_settings.explainable:
            messagebox.showinfo("Explainable", "[EXPLAINABLE] Exiting the application after completing all tasks.")

        messagebox.showinfo("Exit", "Exiting the application.")
        self.root.quit()

from tkinter import Toplevel, Label, Button, Listbox, messagebox, IntVar, Checkbutton, END
import matplotlib.pyplot as plt

class AnalyticsCommand(Command):
    def __init__(self, root):
        self.analytics = AnalyticsFacade()
        self.root = root

    def execute(self):
        """Opens the analytics menu in a new window."""
        analytics_window = Toplevel(self.root)
        analytics_window.title("Analytics")
        analytics_window.geometry("600x400")

        Label(analytics_window, text="Analytics Menu", font=("Helvetica", 16), pady=10).pack()

        options = [
            "Compute Statistics for Selected Models",
            "Visualize Results for Selected Models",
            "Display Results for Selected Models",
            "Exit",
        ]
        options_listbox = Listbox(analytics_window, font=("Helvetica", 12), height=len(options))
        for option in options:
            options_listbox.insert(END, option)
        options_listbox.pack(pady=10, fill="x")

        def handle_selection():
            selected_index = options_listbox.curselection()
            if not selected_index:
                messagebox.showerror("Error", "Please select an option from the menu.")
                return

            choice = selected_index[0] + 1
            if choice == 1:
                self.compute_statistics()
            elif choice == 2:
                self.visualize_results()
            elif choice == 3:
                self.display_results()
            elif choice == 4:
                analytics_window.destroy()
            else:
                messagebox.showerror("Error", "Invalid choice.")

        Button(analytics_window, text="Execute", command=handle_selection).pack(pady=10)

    def compute_statistics(self):
        """Compute and display classification statistics for selected models."""
        model_names = self.select_models()
        if not model_names:
            return
        try:
            results = self.analytics.compute_model_statistics(model_names)
            if results:
                self.display_statistics_in_window(results)
            else:
                messagebox.showinfo("No Results", "No results available to compute statistics.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute statistics: {e}")

    def visualize_results(self):
        """Generate a grouped bar chart for classification results."""
        model_names = self.select_models()
        if not model_names:
            return
        try:
            self.analytics.generate_visualization(model_names)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {e}")

    def display_results(self):
        """Display all classification results for selected models."""
        model_names = self.select_models()
        if not model_names:
            return

        results = self.analytics.load_results_from_csv()
        if not results:
            messagebox.showinfo("No Results", "No results available to display.")
            return

        self.display_results_in_window(results, model_names)

    def select_models(self):
        """Prompt the user to select models for analytics."""
        model_window = Toplevel(self.root)
        model_window.title("Select Models")
        model_window.geometry("300x300")

        Label(model_window, text="Select Models for Analytics", font=("Helvetica", 12)).pack(pady=10)

        model_vars = {}
        models = ["HGBC", "SVM", "CatBoost_Model", "NaiveBayes_Model", "KNN_Model"]
        for model in models:
            var = IntVar(value=0)
            Checkbutton(model_window, text=model, variable=var).pack(anchor="w", padx=20)
            model_vars[model] = var

        selected_models = []

        def save_selection():
            nonlocal selected_models
            selected_models = [model for model, var in model_vars.items() if var.get()]
            model_window.destroy()

        Button(model_window, text="Save", command=save_selection).pack(pady=10)
        model_window.wait_window()

        if not selected_models:
            messagebox.showinfo("No Models Selected", "No models were selected for analytics.")
        return selected_models

    def display_statistics_in_window(self, statistics):
        """Display statistics in a new window."""
        stats_window = Toplevel(self.root)
        stats_window.title("Statistics")
        stats_window.geometry("600x400")

        Label(stats_window, text="Classification Statistics", font=("Helvetica", 16), pady=10).pack()

        listbox = Listbox(stats_window, font=("Helvetica", 12), width=80, height=20)
        for type_label, counts in statistics.items():
            listbox.insert(END, f"\n{type_label}:")
            for category, percentage in counts.items():
                listbox.insert(END, f"  {category}: {percentage:.2f}%")
        listbox.pack(pady=10)

        Button(stats_window, text="Close", command=stats_window.destroy).pack(pady=10)

    def display_results_in_window(self, results, model_names):
        """Display results in a new window."""
        results_window = Toplevel(self.root)
        results_window.title("Classification Results")
        results_window.geometry("700x400")

        Label(results_window, text="Classification Results", font=("Helvetica", 16), pady=10).pack()

        listbox = Listbox(results_window, font=("Helvetica", 12), width=100, height=20)
        filtered_results = self.analytics.filter_results_by_models(results, model_names)
        for result in filtered_results:
            listbox.insert(END, str(result))
        listbox.pack(pady=10)

        Button(results_window, text="Close", command=results_window.destroy).pack(pady=10)
