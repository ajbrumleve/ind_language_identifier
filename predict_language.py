import pickle

import chardet
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import re

data_ind = pd.read_csv("static/Language Detection_ind.csv", encoding_errors="replace", delimiter="\t")

data = data_ind

ngram_files = {"Alas": 'static/Language Detection_btz.csv', "Indonesian": 'static/Language Detection_ind_only.csv',
               "English": 'static/Language Detection_eng.csv'}


def save_model(model: Tuple[MultinomialNB, CountVectorizer, LabelEncoder, Dict[str,List[str]]], file_out: str) -> None:
    pickle.dump(model, open(file_out, 'wb'))


def build_model(show_conf_matrix=False):
    print("Beginning to train language recognition model")
    models = {}
    accuracies = []
    conv_table = pd.DataFrame(columns=["data_size", "accuracy", "f1"])


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
    try:
        X = cv.fit_transform(data_list).toarray()
    except MemoryError:
        X = cv.fit_transform(data_list)
    X.shape  # (10337, 39419)

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    model = MultinomialNB()

    # Define the number of folds for cross-validation
    n_folds = 5

    # Define the cross-validation strategy
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Perform cross-validation and calculate mean scores
    precision_scores = cross_val_score(model, X, y, cv=kf, scoring='precision_macro')
    recall_scores = cross_val_score(model, X, y, cv=kf, scoring='recall_macro')
    accuracy_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=kf, scoring='f1_macro')
    # Perform cross-validation and get predicted labels
    y_pred = cross_val_predict(model, X, y, cv=kf)

    report = classification_report(y, y_pred, digits=5)

    # Print the classification report
    print("Classification Report:")
    print(report)
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

    # model.fit(x_train, y_train)
    #
    # y_pred = model.predict(x_test)
    #
    # pr = precision_score(y_test, y_pred, average=None)[0]
    # rc = recall_score(y_test, y_pred, average=None)[1]
    # ac = accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # f1 = 2 * (pr * rc) / (pr + rc)
    # item = {"data_size": size, "accuracy": ac, "f1": f1}
    # item_df = pd.DataFrame(item, index=[iteration])
    # conv_table = conv_table.append(item_df)
    # models[iteration] = {}
    # models[iteration]["cv"] = cv
    # models[iteration]["model"] = model
    # models[iteration]["pr"] = pr
    # models[iteration]["ac"] = ac
    # models[iteration]["cm"] = cm
    # models[iteration]["f1"] = f1
    # models[iteration]["le"] = le
    # accuracies.append(ac)
    # print(
    #     f"Finished training language recognition model iteration {iteration}. Accuracy is {ac * 100}%. Precision is {pr}. "
    #     f"Recall is {rc}, f1 is {f1}.")

    #
    # print(
    #     f"Iteration {highest_accuracy_iteration} chose. Accuracy is {ac * 100}%. Precision is {pr}. "
    #     f"Recall is {rc}, f1 is {f1}.")

    if show_conf_matrix:
        plt.figure(figsize=(15, 10))
        sns.heatmap(cm, annot=True)
        plt.show()

    print("Creating trigram lists")
    ngram_lists = {}
    for key in ngram_files.keys():
        approved_trigrams = get_grams(key)
        ngram_lists[key] = approved_trigrams
    print("Trigram lists created")
    model.fit(X,y)
    save_model((model, cv, le, ngram_lists), "models/lang_id.mdl")
    return model, cv, le, ngram_lists


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

    list_sentences = pd.read_csv(ngram_files[language], encoding_errors="replace")
    new_approved_trigrams = create_n_grams_from_list(list_sentences['text'], 3)
    return new_approved_trigrams


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
    try:
        filename = "models/lang_id.mdl"
        model, cv, le, ngram_lists = pickle.load(open(filename, 'rb'))
    except:
        new_model = build_model()
        model, cv, le, ngram_lists = new_model

    x = cv.transform([input_text]).toarray()  # converting text to bag of words model (Vector)
    lang = model.predict(x)  # predicting the language
    lang_probs = model.predict_proba(x)
    lang_prob_dict = {}
    for i in range(len(le.classes_)):
        lang_prob_dict[le.classes_[i]] = f"{round(lang_probs[0][i] * 100, 1)}%"
    print(lang_prob_dict)
    lang = le.inverse_transform(lang)  # finding the language corresponding the predicted value
    gram_list = ngram_lists[lang[0]]
    input_familiarity = familiarity(input_text, 3, gram_list)
    print(input_familiarity)
    if input_familiarity > .75:
        print(lang[0], input_familiarity, lang_prob_dict)
        return lang[0], input_familiarity, lang_prob_dict
    else:
        print("language not recognized. Please use Indonesian, Alas, or English","","")
        return "unknown language",input_familiarity,lang_prob_dict
