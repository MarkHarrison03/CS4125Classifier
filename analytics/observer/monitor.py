import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Define the function to call when the file is updated
def on_file_update():
    print("The CSV file has been updated!")

# Custom event handler
class MyHandler(FileSystemEventHandler):
    def __init__(self, file_to_watch, callback):
        self.file_to_watch = os.path.abspath(file_to_watch)  # Ensure absolute path
        self.callback = callback

    def on_modified(self, event):
        # Check if the event is for the target file
        if os.path.abspath(event.src_path) == self.file_to_watch:
            print(f"Detected change in {self.file_to_watch}")
            self.callback()

# Main function to set up the observer
def monitor_csv(file_path):
    file_path = os.path.abspath(file_path)  # Ensure absolute path
    directory = os.path.dirname(file_path)  # Extract the directory

    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"[ERROR] File not found: {file_path}")

    # Set up the observer
    event_handler = MyHandler(file_path, on_file_update)
    observer = Observer()
    observer.schedule(event_handler, path=directory, recursive=False)
    observer.start()

    print(f"Monitoring {file_path} for changes...")
    try:
        while True:
            time.sleep(1)  # Keep the script running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Specify the path to your CSV file
csv_file_path = "../../classification_results.csv"

# Start monitoring
if __name__ == "__main__":
    try:
        monitor_csv(csv_file_path)
    except FileNotFoundError as e:
        print(e)
