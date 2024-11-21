import SVMModel
import HDBCModel
from userSettings import userSettings

## user input :
## subject line
## body 
## 
## Strategy :
##    Preprocessing: Translation, Noise removal 
##    Classification: model choice, level of classification (1, 2, or 3 categories , each of increasing specificity)
##    Output: Verbose, Explainable
def get_model_choice():
    configuration = userSettings()
    print("Choose a model for classification. If selecting multiple, separate your choices with commas:")
    print("1. HDBC Model")
    print("2. SVM Model")
    choiceModel = input("Enter your choice e.g. 1 \n").strip()
    model = None
    if choiceModel:
        choices = choiceModel.split(",")
        for choice in choices:
            if choice.strip() == "1":
                model = "HDBC"
            if choice.strip() == "2":
                model = "SVM"
    
    print("Choose settings for Preprocessing. If selecting multiple, separate your choices with commas:")
    print("1. Automatic text translation")
    print("2. Noise Removal")
    choicePre = input("Enter your choice e.g. 1,2 \n").strip()
    
    translate = False
    noiseRemoval = False
    
    if choicePre:
        choices = choicePre.split(",")
        for choice in choices:
            if choice.strip() == "1":
                translate = True
            if choice.strip() == "2":
                noiseRemoval = True
                
    print("Choose settings for Preprocessing. If selecting multiple, separate your choices with commas:")
    print("1. Verbose output")
    print("2. Explainable AI")
    choicePost = input("Enter your choice e.g. 2 \n").strip()
    verbose = False
    explainable = False
    
    if choicePost:
        choices = choicePost.split(",")
        for choice in choices:
            if choice.strip() == "1":
                verbose = True
            if choice.strip() == "2":
                explainable = True
                          
    configuration.update_settings( model, translate, noiseRemoval, verbose, explainable)
    print(configuration)
    
    return 
get_model_choice()
subject = input("Please enter subject line: ")
email = input("Please enter email body text: ")


"""if choice == "1":
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
"""