import csv
import os
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


class AnalyticsFacade:
    def __init__(self, results_file="classification_results.csv"):
        self.results_file = results_file

    def load_results_from_csv(self):
        """
        Loads classification results from the CSV file into a list of dictionaries.
        """
        try:
            with open(self.results_file, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                results = [row for row in reader]
                print(f"Loaded {len(results)} results from {self.results_file}.")
                print(f"Results loaded: {results}")  # Debugging
                return results
        except FileNotFoundError:
            print(f"{self.results_file} not found. No results available.")
            return []
        except Exception as e:
            print(f"Error loading results from {self.results_file}: {e}")
            return []

    def display_menu(self):
        """Displays the analytics menu and returns the user's choice."""
        print("\n=== Email Classification Analytics ===")
        print("1. Compute Statistics for Selected Models")
        print("2. Visualize Results for Selected Models")
        print("3. Display Results for Selected Models")
        print("4. Exit")
        return input("Enter your choice (1-4): ").strip()

    def get_model_choices(self):
        """Lets the user choose which model(s) to analyze."""
        available_models = {
            1: "HGBC",
            2: "SVM",
            3: "CatBoost_Model",
            4: "NaiveBayes_Model",
            5: "KNN_Model"
        }

        print("\nAvailable Models:")
        for idx, model_name in available_models.items():
            print(f"{idx}. {model_name}")

        choices = input("Select models (e.g., 1,2): ").strip()
        selected_models = []

        try:
            for choice in choices.split(","):
                choice = int(choice.strip())
                if choice in available_models:
                    selected_models.append(available_models[choice])
                else:
                    print(f"Invalid selection: {choice}")
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

        if not selected_models:
            print("No valid models selected. Please try again.")
            return self.get_model_choices()

        return selected_models

    def compute_grouped_counts(self, model_names, results):
        """
        Counts the classification results grouped by type (Type 1, Type 2, Type 3, Type 4) for selected models.
        """
        grouped_counts = {
            "Type 1": Counter(),
            "Type 2": Counter(),
            "Type 3": Counter(),
            "Type 4": Counter(),
        }

        for result in results:
            for model_name in model_names:
                for type_index, type_label in enumerate(["Type 1", "Type 2", "Type 3", "Type 4"], start=1):
                    column_key = f"{model_name}_Type{type_index}"  # e.g., HGBC_Type1, HGBC_Type2, etc.
                    if column_key in result and result[column_key] != "N/A":
                        # Split and normalize classifications
                        categories = result[column_key].split(",") if result[column_key] else []
                        for category in categories:
                            grouped_counts[type_label][category.strip()] += 1

        return grouped_counts

    def compute_model_statistics(self, model_names):
        """
        Calculates the percentage distribution of classifications for all types (Type 1, Type 2, Type 3, Type 4).
        """
        results = self.load_results_from_csv()
        if not results:
            print("No results available to compute statistics.")
            return

        grouped_counts = self.compute_grouped_counts(model_names, results)
        total_by_type = {type_label: sum(counts.values()) for type_label, counts in grouped_counts.items()}

        print("\n=== Classification Statistics ===")
        for type_label, counts in grouped_counts.items():
            total = total_by_type[type_label]
            if total > 0:
                print(f"\n{type_label}:")
                for classification, count in counts.items():
                    percentage = (count / total) * 100
                    print(f"  {classification}: {percentage:.2f}%")
            else:
                print(f"\n{type_label}: No results available.")

    def generate_visualization(self, model_names):
        """
        Creates a grouped bar chart for the classification results.
        """
        results = self.load_results_from_csv()
        if not results:
            print("No results available to visualize.")
            return

        grouped_counts = self.compute_grouped_counts(model_names, results)
        if not grouped_counts:
            print("No valid classifications available to visualize.")
            return

        categories = list(grouped_counts.keys())
        counts = list(grouped_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(categories, counts, color="skyblue")
        plt.xlabel("Classifications")
        plt.ylabel("Counts")
        plt.title("Classification Results Visualization")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def display_results(self, model_names):
        """
        Displays all classification results for the selected models.
        """
        results = self.load_results_from_csv()
        if not results:
            print("No results available to display.")
            return

        print("\n=== Classification Results ===")
        for result in results:
            filtered_result = {
                key: result[key]
                for key in ["subject", "email"] + model_names
                if key in result and result[key] not in ["", "N/A"]
            }
            if filtered_result:
                print(filtered_result)
            else:
                print("No relevant results for the selected models.")