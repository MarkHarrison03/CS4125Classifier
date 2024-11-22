import ast
from collections import Counter
from models.modelClass.HGBCModel import HGBCModel
from models.modelClass.SVMModel import SVMModel
import numpy as np
import matplotlib.pyplot as plt


class AnalyticsFacade:
    def __init__(self):
        self.results = []

    def display_menu(self):
        """Displays the main menu and returns the user's choice."""
        print("\n=== Email Classification Analytics ===")
        print("1. Add Classification Results")
        print("2. Compute Statistics for Selected Models")
        print("3. Visualize Results for Selected Models")
        print("4. Display Results for Selected Models")
        print("5. Exit")
        return input("Enter your choice (1-5): ")

    def get_model_choices(self):
        """Allows the user to select one or more models for processing."""
        available_models = {
            1: "HDBC_Model",
            2: "SVM_Model",
            # Add new models here as needed
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

    def parse_classification(self, classification_result):
        """Parses raw classification results into a list of types."""
        try:
            if isinstance(classification_result, np.ndarray):
                classification_result = classification_result.tolist()

            if isinstance(classification_result, list) and len(classification_result) > 0:
                first_element = classification_result[0]
                if isinstance(first_element, list):
                    return first_element
                elif isinstance(first_element, str):
                    return first_element.strip("[]").split()
            return None
        except Exception as e:
            print(f"Failed to parse classification result: {e}")
            return None

    def add_classification_result(self, subject, email):
        """Adds classification results for the given subject and email."""
        classification_hdbc = HGBCModel.categorize(subject, email)
        classification_svm = SVMModel.categorize(subject, email)

        parsed_hdbc = self.parse_classification(classification_hdbc)
        parsed_svm = self.parse_classification(classification_svm)

        if parsed_hdbc and parsed_svm:
            self.results.append({
                'subject': subject,
                'email': email,
                'HDBC_Model': parsed_hdbc,
                'SVM_Model': parsed_svm,
            })
            print("Classification results added successfully!")
        else:
            print("Failed to parse classification results.")

    def compute_grouped_counts(self, model_names):
        """Computes grouped counts for Types 2, 3, and 4."""
        grouped_counts = {
            "Type 2": Counter(),
            "Type 3": Counter(),
            "Type 4": Counter(),
        }

        for result in self.results:
            for model_name in model_names:
                if model_name in result:
                    classification = result[model_name]
                    grouped_counts["Type 2"][classification[0]] += 1
                    grouped_counts["Type 3"][classification[1]] += 1
                    grouped_counts["Type 4"][classification[2]] += 1

        return grouped_counts

    def compute_model_statistics(self, model_names):
        """Calculates percentages for each category in Types 2, 3, and 4."""
        grouped_counts = self.compute_grouped_counts(model_names)

        statistics = {}
        for type_label, counts in grouped_counts.items():
            total = sum(counts.values())
            if total > 0:
                statistics[type_label] = {category: (count / total * 100) for category, count in counts.items()}

        return statistics

    def generate_visualization(self, model_names):
        """Generates a grouped bar chart for Types 2, 3, and 4."""
        grouped_counts = self.compute_grouped_counts(model_names)

        type_labels = list(grouped_counts.keys())
        unique_categories = {
            type_label: sorted(count.keys()) for type_label, count in grouped_counts.items()
        }

        bar_heights = {
            type_label: [grouped_counts[type_label].get(category, 0) for category in unique_categories[type_label]]
            for type_label in type_labels
        }

        bar_width = 0.2
        x_positions = np.arange(len(type_labels))
        fig, ax = plt.subplots(figsize=(12, 6))

        for i, type_label in enumerate(type_labels):
            categories = unique_categories[type_label]
            for j, category in enumerate(categories):
                bar_position = x_positions[i] + j * bar_width
                ax.bar(bar_position, bar_heights[type_label][j], bar_width, label=f"{type_label}: {category}")

        ax.set_xlabel("Types")
        ax.set_ylabel("Count")
        ax.set_title("Classification Results Grouped by Type")
        ax.set_xticks(x_positions + (len(bar_heights["Type 2"]) - 1) * bar_width / 2)
        ax.set_xticklabels(type_labels)
        ax.legend(title="Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    analytics = AnalyticsFacade()

    while True:
        choice = analytics.display_menu()
        if choice == "1":
            subject = input("Enter email subject: ")
            email = input("Enter email body: ")
            analytics.add_classification_result(subject, email)

        elif choice == "2":
            model_names = analytics.get_model_choices()
            stats = analytics.compute_model_statistics(model_names)
            for type_label, percentages in stats.items():
                print(f"\n{type_label}:")
                for category, percentage in percentages.items():
                    print(f"  {category}: {percentage:.2f}%")

        elif choice == "3":
            model_names = analytics.get_model_choices()
            analytics.generate_visualization(model_names)

        elif choice == "4":
            model_names = analytics.get_model_choices()
            print("\n=== All Classification Results for Selected Models ===")
            for result in analytics.results:
                for model_name in model_names:
                    if model_name in result:
                        print(f"Model: {model_name}")
                        print(result[model_name])

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please try again.")
