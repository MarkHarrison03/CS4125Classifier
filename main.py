import HDBCModel
subject = input("Please enter subject line:")
email = input("Please enter email body text:")

classifier_type = ("Please enter your classifier type (current available classifiers: HDBC)")


model = HDBCModel
classification = HDBCModel.categorize(subject, email)
print(classification)
