import models.modelClass.SVMModel as SVMModel
import models.modelClass.HDBCModel as HDBCModel

def get_model_choice():
    print("Choose a model for classification:")
    print("1. HDBC Model")
    print("2. SVM Model")
    print("3. Run Both Models")
    choice = input("Enter your choice (1/2/3): ")
    return choice

subject = input("Please enter subject line: ")
email = input("Please enter email body text: ")

choice = get_model_choice()

if choice == "1":
    try:
        classification_hdbc = HDBCModel.categorize(subject, email)
        print("\nHDBC Model Classification:", classification_hdbc)
    except ValueError as e:
        print(f"\nHDBC Model Error: {e}")

elif choice == "2":
    try:
        classification_svm = SVMModel.categorize(subject, email)
        print("\nSVM Model Classification:", classification_svm)
    except ValueError as e:
        print(f"\nSVM Model Error: {e}")

elif choice == "3":
    try:
        classification_hdbc = HDBCModel.categorize(subject, email)
        print("\nHDBC Model Classification:", classification_hdbc)
    except ValueError as e:
        print(f"\nHDBC Model Error: {e}")

    try:
        classification_svm = SVMModel.categorize(subject, email)
        print("\nSVM Model Classification:", classification_svm)
    except ValueError as e:
        print(f"\nSVM Model Error: {e}")

else:
    print("\nInvalid choice. Please run the program again and select 1, 2, or 3.")
