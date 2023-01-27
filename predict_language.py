import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import re

print("Beginning to train language recognition model")
# data=pd.read_csv("D:/C Downloads/archive (1)/Language Detection.csv")
data_ind = pd.read_csv("static/Language Detection_ind.csv")
# data_ind=pd.read_csv("D:/C Downloads/archive (1)/Language Detection_ind_with_fb.csv")
# data=np.array_split(data, 2)[0]
# data=pd.concat([data,data_ind])
data = data_ind
conv_table = pd.DataFrame(columns=["data_size", "accuracy", "f1"])
ngram_files = {"Alas": 'static/Language Detection_btz.csv', "Indonesian": 'static/Language Detection_ind_only.csv',
               "English": 'static/Language Detection_eng.csv'}
for iteration in range(10):
    data = data_ind
    size = round((iteration + 1) * len(data) / 10)
    data = data.sample(n=size)
    X = data["text"]
    y = data["language"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    data_list = []
    # iterating through all the text
    for text in X:
        # removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)

    cv = CountVectorizer()
    X = cv.fit_transform(data_list).toarray()
    X.shape  # (10337, 39419)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model = MultinomialNB()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    pr = precision_score(y_test, y_pred, average=None)[0]
    rc = recall_score(y_test, y_pred, average=None)[1]
    ac = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1 = 2 * (pr * rc) / (pr + rc)
    item = {"data_size": size, "accuracy": ac, "f1": f1}
    item_df = pd.DataFrame(item, index=[iteration])
    conv_table = conv_table.append(item_df)
print("Finished training language recognition model. Accuracy is ", ac * 100, "%. Precision is ", pr, ". Recall is ",
      rc, "f1 is ", f1, ".")


# Accuracy is : 0.9772727272727273


# plt.figure(figsize=(15,10))
# sns.heatmap(cm, annot=True)
# plt.show()


def create_n_grams(string, n, gram_list):
    list_grams = gram_list
    word_len = len(string)
    for i in range(word_len - n + 1):
        gram = string[i:i + n]
        if gram not in list_grams:
            list_grams.append(gram)
    return list_grams


def create_n_grams_from_list(list_of_strings, n):
    gram_list = []
    for new_item in list_of_strings:
        gram_list = create_n_grams(new_item, n, gram_list)
    return gram_list


def get_grams(language):
    exclude_list = [18, 186]

    list_sentences = pd.read_csv(ngram_files[language])
    new_approved_trigrams = create_n_grams_from_list(list_sentences['text'], 3)
    return new_approved_trigrams


print("Creating trigram lists")
ngram_lists = {}
for key in ngram_files.keys():
    approved_trigrams = get_grams(key)
    ngram_lists[key] = approved_trigrams
print("Trigram lists created")


def familiarity(string, n, gram_list):
    list_grams_approved = gram_list
    num_not_approved = 0
    word_len = len(string)
    accuracy = float(0)
    for i in range(word_len - n + 1):
        gram = string[i:i + n]
        if gram not in list_grams_approved:
            num_not_approved = num_not_approved + 1
        accuracy = 1 - (num_not_approved / (i + 1))
    return accuracy


def predict(input_text):
    x = cv.transform([input_text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang_probs = model.predict_proba(x)
    lang = le.inverse_transform(lang)  # finding the language corresponding the the predicted value
    gram_list = ngram_lists[lang[0]]
    if familiarity(input_text, 3, gram_list) > .75:
        return lang[0]
    else:
        return "language_salah"
