import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.multioutput import MultiOutputClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from googletrans import Translator
import joblib



df = pd.read_csv("AppGallery.csv")

# convert the dtype object to unicode string
df['Interaction content'] = df['Interaction content'].values.astype('U')
df['Ticket Summary'] = df['Ticket Summary'].values.astype('U')

df["y1"] = df["Type 1"]
df["y2"] = df["Type 2"]
df["y3"] = df["Type 3"]
df["y4"] = df["Type 4"]
df["x"] = df['Interaction content']

df["y"] = df["y2"]
labels = df[['Type 2', 'Type 3', 'Type 4']]

# remove empty y
df = df.loc[(df["y"] != '') & (~df["y"].isna()),]

temp = df


def trans_to_en(texts):
    translator = Translator()
    for i in range(0, len(texts)):
        texts[i] = translator.translate(texts[i], dest = 'en').text
        print(texts[i])
    return texts
    
    
good_y1 = temp.y1.value_counts()[temp.y1.value_counts() > 10].index

temp = temp.loc[temp.y1.isin(good_y1)]

temp['Interaction content'] = trans_to_en(temp['Interaction content'])
temp['Ticket Summary'] = trans_to_en(temp['Ticket Summary'])

print(temp['Interaction content'])
print(temp['Ticket Summary'])

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
x1 = tfidfconverter.fit_transform(temp["Interaction content"]).toarray()
x2 = tfidfconverter.transform(temp["Ticket Summary"]).toarray()  
X = np.concatenate((x1, x2), axis=1)
print("x1:")
print(x1)
print("x2:")
print(x2)
print("x:")
print(X)

 
#data prep
y = temp[['Type 2', 'Type 3', 'Type 4']]

filtered_temp = temp.dropna(subset=['Type 2', 'Type 3', 'Type 4'])  
filtered_temp = filtered_temp[filtered_temp['Type 2'] != '']  
filtered_temp = filtered_temp[filtered_temp['Type 3'] != '']  
filtered_temp = filtered_temp[filtered_temp['Type 4'] != '']  

y = filtered_temp[['Type 2', 'Type 3', 'Type 4']]  
X = X[filtered_temp.index]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

feature_names_x1 = tfidfconverter.get_feature_names_out(input_features=temp["Interaction content"].values)
feature_names_x2 = tfidfconverter.get_feature_names_out(input_features=temp["Ticket Summary"].values)

feature_names = np.concatenate([feature_names_x1, feature_names_x2])

print(f"Number of features in X_test: {X_test.shape[1]}")
print(f"Number of feature names: {len(feature_names)}")
#model select
classifier = MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=2000, random_state=0))
#training
classifier.fit(X_train, y_train)
#testing
y_pred = classifier.predict(X_test)

#display results
p_result = pd.DataFrame(y_pred)
p_result.columns = y.columns
print(p_result)

accuracies = []
for i in range(y_test.shape[1]):
    accuracy = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
    accuracies.append(accuracy)

print(accuracies)
average_accuracy = sum(accuracies) / len(accuracies)
print(f"Average accuracy: {average_accuracy:.2f}")

explainer = shap.TreeExplainer(classifier.estimators_[0])

values = explainer.shap_values(X_test)

shap.initjs()
shap.summary_plot(values, X_test, feature_names=feature_names)
joblib.dump(tfidfconverter, 'tfidf_vectorizer.pkl')
joblib.dump(classifier, 'HDBCModel.pkl')