import math
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ADVERSARIAL = False
input_files = [
  'static/Language Detection_btz.csv',
  'static/Language Detection_eng.csv',
  'static/Language Detection_ind_only.csv',
]
adversarial_file = [

]

if len(adversarial_file) > 0:
    ADVERSARIAL = True
data=pd.read_csv(input_files[0])
for file in input_files[1:]:
    data=pd.concat([data,pd.read_csv(file)])
# filter for sentences with more than 3 words
data = data[data["text"].str.split().str.len() > 3]
input_data = []
labels = []

if ADVERSARIAL == True:
    adversarial_data = pd.read_csv('static/Language Detection_eng.csv')
    adversarial_input_data = adversarial_data["text"].reset_index(drop=True)
    adversarial_labels = adversarial_data["language"].reset_index(drop=True)
    adversarial_label_map = {"Alas": 0, "Indonesian": 1, "English": 2}
    adversarial_labels = adversarial_labels.replace(adversarial_label_map, regex=True)
    adversarial_X_train, adversarial_X_test, adversarial_y_train, adversarial_y_test = train_test_split(
        adversarial_input_data, adversarial_labels)

else:
    adversarial_data=pd.DataFrame(columns=["text","language"])
    adversarial_X_test=[]
    adversarial_y_test=[]


for label, f in enumerate(input_files):
  print(f"{f} corresponds to label {label}")

  for line in open(f):
    line = line.rstrip().lower()
    if line:
      # remove punctuation
      line = line.translate(str.maketrans('', '', string.punctuation))

      input_data.append(line)
      labels.append(label)

input_data = data["text"].reset_index(drop=True)
labels = data["language"].reset_index(drop=True)
label_map = {"Alas": 0, "Indonesian": 1, "English": 2}
labels = labels.replace(label_map,regex=True)




#create test_train split
X_train, X_test, y_train, y_test = train_test_split(input_data, labels)

X_test = X_test.append(adversarial_X_test)
y_test = y_test.append(adversarial_y_test)

#get unique words
def split_sentence(list_sentences):
    data2=[]
    #loop through data and split each line
    for i in range(len(list_sentences)):
        list_sentences[i]=input_data[i]
        data2.append(list_sentences[i].split())
        data2.append(list_sentences[i].split())
    return data2

def create_n_grams(string, n):
    list_grams = []
    word_len = len(string)
    for i in range(word_len - n + 1):
        gram = string[i:i + n]
        list_grams.append(gram)
    list_grams.append(string[(word_len-n+1):(word_len+1+n)])
    return list_grams

def create_n_grams_from_list(list_of_strings, n):
    gram_list = []
    for item in list_of_strings:
        gram_list = create_n_grams(item, n, gram_list)
    return gram_list

    #assign each uniqe word an integer
int_index = 1
mapping = {}
mapping["<unk>"] = 0
for text in X_train:
    # tokens = text.split()
    tokens= create_n_grams(text,3)
    for token in tokens:
        if token not in mapping:
            mapping[token] = int_index
            int_index += 1
# for item in split_sentence(X_train):
#     for word in item:
#         if word not in mapping.keys():
#             mapping[word] = int_index
#             int_index += 1
X_train_int = []
X_test_int = []
for item in X_train:
    sentence = create_n_grams(item,3)
    line_as_int = [mapping[token] for token in sentence]
    X_train_int.append(line_as_int)

for item in X_test:
    sentence = create_n_grams(item,3)
    line_as_int = [mapping.get(token,0) for token in sentence]
    X_test_int.append(line_as_int)


V = len(mapping)
num_lang = len(input_files)
markov_dict = {}
for i in range(num_lang):
    markov_dict['A{}'.format(str(i))] = np.ones((V, V))
    markov_dict['pi{}'.format(str(i))] = np.ones((V))

A0 = np.ones((V, V))
pi0 = np.ones(V)

A1 = np.ones((V, V))
pi1 = np.ones(V)

A2 = np.ones((V, V))
pi2 = np.ones(V)

# compute counts for A and pi
def compute_counts(text_as_int, A, pi):
    for tokens in text_as_int:
        last_idx = None
        for idx in tokens:
            if last_idx is None:
                # it's the first word in a sentence
                pi[idx] += 1
            else:
                # the last word exists, so count a transition
                A[last_idx, idx] += 1
            # update last idx
            last_idx = idx
total = len(y_train)
for i in range(num_lang):


    compute_counts([t for t, y in zip(X_train_int, y_train) if y == i], markov_dict['A{}'.format(str(i))], markov_dict['pi{}'.format(str(i))])
    A = markov_dict['A{}'.format(str(i))]
    pi = markov_dict['pi{}'.format(str(i))]
    markov_dict['A{}'.format(str(i))] /= A.sum(axis=1, keepdims=True)
    markov_dict['pi{}'.format(str(i))] /= pi.sum()
    markov_dict['logA{}'.format(str(i))] = np.log(markov_dict['A{}'.format(str(i))])
    markov_dict['logpi{}'.format(str(i))] = np.log(markov_dict['pi{}'.format(str(i))])
    markov_dict["count{}".format(str(i))] = count = sum(y == i for y in y_train)
    markov_dict["p{}".format(str(i))] = count / total
    markov_dict["logp{}".format(str(i))] = np.log(markov_dict["p{}".format(str(i))])


compute_counts([t for t, y in zip(X_train_int, y_train) if y == 0], A0, pi0)
compute_counts([t for t, y in zip(X_train_int, y_train) if y == 1], A1, pi1)
compute_counts([t for t, y in zip(X_train_int, y_train) if y == 2], A2, pi2)

A0 /= A0.sum(axis=1, keepdims=True)
pi0 /= pi0.sum()

A1 /= A1.sum(axis=1, keepdims=True)
pi1 /= pi1.sum()

A2 /= A2.sum(axis=1, keepdims=True)
pi2 /= pi2.sum()

logA0 = np.log(A0)
logA1 = np.log(A1)
logA2 = np.log(A2)

logpi0 = np.log(pi0)
logpi1 = np.log(pi1)
logpi2 = np.log(pi2)



count0 = sum(y == 0 for y in y_train)
count1 = sum(y == 1 for y in y_train)
count2 = sum(y == 2 for y in y_train)
total = len(y_train)
p0 = count0 / total
p1 = count1 / total
p2 = count1 / total
logp0 = np.log(p0)
logp1 = np.log(p1)
logp2 = np.log(p2)
p0, p1, p2

# build a classifier
class Classifier:
    def __init__(self, logAs, logpis, logpriors):
        self.logAs = logAs
        self.logpis = logpis
        self.logpriors = logpriors
        self.K=len(logpriors)

    def compute_prob(self, input_, class_):
        logA = self.logAs[class_]
        logpi = self.logpis[class_]

        last_idx = None
        logprob = 0
        for idx in input_:
            if last_idx is None:
                logprob += logpi[idx]
            else:
                logprob += logA[last_idx, idx]
            last_idx = idx
        return logprob

    def predict(self, inputs):
        predictions = np.zeros(len(inputs))
        pred_values = np.zeros(len(inputs))
        pred_diffs = np.zeros(len(inputs))
        for i, input_ in enumerate(inputs):
            posteriors = [self.compute_prob(input_, c) + self.logpriors[c] for c in range(self.K)]
            sorted_post = posteriors.copy()
            sorted_post.sort(reverse=True)
            pred = np.argmax(posteriors)
            pred_diff = sorted_post[0] - sorted_post[1]
            # if pred_diff < 25:
            #     predictions[i] = -1
            # else:
            #     predictions[i] = pred
            predictions[i] = pred
            pred_diffs[i] = pred_diff
            pred_values[i] = sorted_post[0]

        return predictions, pred_diffs, pred_values
logAs=[]
logpis=[]
logps=[]
for i in range(len(input_files)):
    logAs.append(markov_dict["logA{}".format(str(i))])
    logpis.append(markov_dict["logpi{}".format(str(i))])
    logps.append(markov_dict["logp{}".format(str(i))])

clf = Classifier(logAs,logpis,logps)

Ptrain = clf.predict(X_train_int)[0]
print(f"Train acc: {np.mean(Ptrain == y_train)}")

Ptest = clf.predict(X_test_int)[0]
print(f"Test acc: {np.mean(Ptest == y_test)}")

from sklearn.metrics import confusion_matrix, f1_score
cm = confusion_matrix(y_train, Ptrain)
cm

cm_test = confusion_matrix(y_test, Ptest)
cm_test

print(f1_score(y_train,Ptrain,average="weighted"))
print(f1_score(y_test,Ptest,average="weighted"))

results_df = pd.concat([y_test.reset_index(), pd.Series(Ptest).astype(int), pd.Series(clf.predict(X_test_int)[1]), pd.Series(clf.predict(X_test_int)[2])], axis=1)
results_incorrect = results_df[results_df['language'] != results_df[0]].reset_index(drop=True)

sentences = X_test[results_incorrect["index"]].reset_index(drop=True)
results_incorrect = pd.concat([results_incorrect,sentences],axis=1,ignore_index=True)
results_incorrect = results_incorrect.rename(columns={0:"index",1:"language",2: 'predicted',3:"sentence"})

with_vals=results_df.sort_values([2])
with_vals_approve = with_vals[((with_vals[2]>-900) & (with_vals[1]>25)) | (with_vals[1]>70)]
with_vals_flag = with_vals[(with_vals[1]<25)| ((with_vals[2]<-900) & (with_vals[1]<70))]

def decode(label_int):
    prediction = [i for i in label_map if label_map[i] == label_int][0]
    return prediction
def process_input(input_text):
    input_ngrams = create_n_grams(input_text, 3)
    input_line_as_int = [mapping.get(input_token, 0) for input_token in input_ngrams]
    prediction = decode(int(clf.predict([input_line_as_int])[0][0]))
    return prediction

