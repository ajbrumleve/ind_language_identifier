import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

from utils import get_domain_class_report

data = pd.read_csv("D:/C Downloads/indo_alas_class.csv", encoding_errors="replace")
X = data["Alas"]
y = data["Domain"]

le = LabelEncoder()
y = le.fit_transform(y)

data_list = []
# iterating through all the text
for text in X:
    # removing the symbols and numbers
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = re.sub(r'r', 'kh', text)
    # converting the text to lower case
    text = text.lower()
    # appending to data_list
    data_list.append(text)

cv = CountVectorizer()
try:
    X = cv.fit_transform(data_list)
except MemoryError:
    X = cv.fit_transform(data_list)
X.shape  # (10337, 39419)

model = MultinomialNB(alpha=.6)

n_folds = 5

# Define the cross-validation strategy
kf = KFold(n_splits=n_folds, shuffle=True)

# Perform cross-validation and calculate mean scores
precision_scores = cross_val_score(model, X, y, cv=kf, scoring='precision_macro',verbose=True)
recall_scores = cross_val_score(model, X, y, cv=kf, scoring='recall_macro')
accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
# Perform cross-validation and get predicted labels
y_pred = cross_val_predict(model, X, y, cv=kf)


get_domain_class_report(y, y_pred)


# Calculate the confusion matrix
cm = confusion_matrix(y, y_pred)

# Calculate the mean of the scores
mean_precision = precision_scores.mean()
mean_recall = recall_scores.mean()
mean_accuracy = accuracy_scores.mean()
mean_f1 = f1_scores.mean()

# Print the mean scores
print("Mean Precision:", mean_precision)
print("Mean Recall:", mean_recall)
print("Mean Accuracy:", mean_accuracy)
print("Mean F1:", mean_f1)


plt.figure(figsize=(15, 10))
sns.heatmap(cm, annot=True)
plt.show()

