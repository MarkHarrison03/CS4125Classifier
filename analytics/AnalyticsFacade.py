import csv
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("MacOSX")

class AnalyticsFacade:
    def __init__(self, results_file="classification_results.csv"):
        self.results_file = results_file
        self.model_prefix_map = {
            "HGBC": "HGBC",
            "SVM": "SVM",
            "CatBoost_Model": "CB",
            "NaiveBayes_Model": "NB",
            "KNN_Model": "KNN"
        }

    def load_results_from_csv(self):
        try:
            with open(self.results_file, mode="r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                results = [row for row in reader]
                print(f"Loaded {len(results)} results from {self.results_file}.")
                return results
        except FileNotFoundError:
            print(f"{self.results_file} not found. No results available.")
            return []
        except Exception as e:
            print(f"Error loading results from {self.results_file}: {e}")
            return []

    def filter_results_by_models(self, results, model_names):
        filtered_results = []
        for result in results:
            filtered_row = {"subject": result.get("subject", ""), "email": result.get("email", "")}
            valid_row = False
            for model_name in model_names:
                prefix = self.model_prefix_map.get(model_name, model_name)
                for i in range(1, 5):
                    column_key = f"{prefix}_Type{i}"
                    value = result.get(column_key, "").strip() if result.get(column_key) else ""
                    if value and value.lower() not in [model_name.lower(), prefix.lower(), ""]:
                        filtered_row[column_key] = value
                        valid_row = True
                    else:
                        filtered_row[column_key] = ""
            if valid_row:
                filtered_results.append(filtered_row)
        print(f"Filtered Results: {filtered_results}")
        return filtered_results

    def compute_grouped_counts(self, model_names, results):
        filtered_results = self.filter_results_by_models(results, model_names)
        grouped_counts = {f"Type {i}": Counter() for i in range(1, 5)}
        for result in filtered_results:
            for model_name in model_names:
                prefix = self.model_prefix_map.get(model_name, model_name)
                for type_index, type_label in enumerate(["Type 1", "Type 2", "Type 3", "Type 4"], start=1):
                    column_key = f"{prefix}_Type{type_index}"
                    value = result.get(column_key, "").strip()
                    if value and value.lower() not in [model_name.lower(), prefix.lower(), ""]:
                        categories = value.split(",")
                        for category in categories:
                            grouped_counts[type_label][category.strip()] += 1
        print(f"Grouped Counts: {grouped_counts}")
        return grouped_counts

    def compute_model_statistics(self, model_names):
        """
        Calculates the percentage distribution of classifications for each type.
        """
        results = self.load_results_from_csv()
        if not results:
            print("No results available to compute statistics.")
            return {}

        grouped_counts = self.compute_grouped_counts(model_names, results)
        total_by_type = {type_label: sum(counts.values()) for type_label, counts in grouped_counts.items()}

        stats = {}
        for type_label, counts in grouped_counts.items():
            total = total_by_type[type_label]
            stats[type_label] = {}
            if total > 0:
                for classification, count in counts.items():
                    percentage = (count / total) * 100
                    stats[type_label][classification] = percentage
            else:
                stats[type_label] = {"No data": 0.0}

        print(f"Computed Statistics: {stats}")
        return stats

    def generate_visualization(self, model_names):
        results = self.load_results_from_csv()
        if not results:
            print("No results available to visualize.")
            return
        grouped_counts = self.compute_grouped_counts(model_names, results)
        if not any(count for count in grouped_counts.values()):
            print("No valid classifications available to visualize.")
            return
        categories, counts = [], []
        for type_label, counter in grouped_counts.items():
            for category, count in counter.items():
                if count > 0:
                    categories.append(f"{type_label} - {category}")
                    counts.append(count)
        if not categories or not counts:
            print("No valid data to visualize.")
            return
        plt.figure(figsize=(12, 8))
        plt.bar(categories, counts, color="skyblue")
        plt.xlabel("Classifications")
        plt.ylabel("Counts")
        plt.title("Classification Results Visualization")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        try:
            plt.show()
        except Exception as e:
            plt.savefig("visualization.png")
            print(f"Visualization saved as 'visualization.png' due to error: {e}")
