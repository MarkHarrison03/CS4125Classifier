import NaiveBayes
import SVMModel
import HDBCModel
import NearestNeighbor


def get_model_choice():
    print("Choose a model for classification:")
    print("1. HDBC Model")
    print("2. SVM Model")
    print("3. Naive Bayes Model")
    print("4. Nearest Neighbors Model")
    print("this will be the other model :)")
    print("6. Run All Models")
    choice = input("Enter your choice (1/2/3/4/5/6): ")
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
        classification_nb = NaiveBayes.categorize(subject, email)
        print("\nNaive Bayes Model Classification:", classification_nb)
    except ValueError as e:
        print(f"\nNaive Bayes Model Error: {e}")

elif choice == "4":
    try:
        classification_nn = NearestNeighbor.categorize(subject, email)
        print("\nNearest Neighbors Model Classification:", classification_nn)
    except ValueError as e:
        print(f"\nNearest Neighbors Model Error: {e}")

elif choice == "6":
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

    try:
        classification_nb = NaiveBayes.categorize(subject, email)
        print("\nNaive Bayes Model Classification:", classification_nb)
    except ValueError as e:
        print(f"\nNaive Bayes Model Error: {e}")

    try:
        classification_nn = NearestNeighbor.categorize(subject, email)
        print("\nNearest Neighbors Model Classification:", classification_nn)
    except ValueError as e:
        print(f"\nNearest Neighbors Model Error: {e}")

else:
    print("\nInvalid choice. Please run the program again and select a valid option.")
