import math

import numpy as np
import os


def load_vocabulary(filename):
    """Load the vocabulary and returns it.

    The return value is a dictionary mapping words to numerical
indices.

    """
    f = open(filename)
    n = 0
    voc = {}
    for w in f.read().split():
        voc[w] = n
        n += 1
    f.close()
    return voc


def remove_punctuation(text):
    """Replace punctuation symbols with spaces."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    for p in punct:
        text = text.replace(p, " ")
    return text


'''
def read_document(filename, voc):
    """Read a document and return its BoW representation."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    text = remove_punctuation(text.lower())
    # Start with all zeros
    bow = np.zeros(len(voc))
    for w in text.split():
        # If the word is the vocabulary...
        if w in voc:
            # ...increment the proper counter.
            index = voc[w]
            bow[index] += 1
    return bow
'''


def read_document(filename, voc, label):
    """Read a document and return its BoW representation."""
    documents = []
    labels = []
    bow = np.zeros(len(voc))

    counter = np.zeros(len(voc))

    with open(filename, encoding="utf8") as f:
        for line in f:
            bow = np.zeros(len(voc))
            text = remove_punctuation(line.lower())
            for w in text.split():
                if w in voc:
                    index = voc[w]
                    bow[index] += 1
                    counter[index] += 1
            documents.append(bow)
            labels.append(label)
    return documents, labels, counter


# The script compute the BoW representation of all the training
# documents.  This need to be extended to compute similar
# representations for the validation and the test set.
voc = load_vocabulary("vocabulary.txt")
documents = []
labels = []
counter = []

documents, labels, counter = read_document("232306-clickbait/clickbait_train.txt", voc, 1)
print(" ")
documents1, labels1, counter1 = read_document("232306-clickbait/non_clickbait_train.txt", voc, 0)
print(" ")
for bow in documents:
    for i in range(len(bow)):
        try:
            if (counter[i]+counter1[i]) > 0:
                # bow[i] = bow[i] * (2 * (abs((counter[i]/(counter[i]+counter1[i])) - 0.5)))
                bow[i] = bow[i] * (pow(((counter[i]/(counter[i]+counter1[i])) - 0.5),2)) * 4
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 4)) * 16
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 12)) * 4096
                # x = (counter[i] / (counter[i] + counter1[i]))
                # a = (pow((x - 0.5), 4))
                # bow[i] = bow[i] * a * ((600 * math.exp(x)) / (100 + math.exp(x)))
            else:
                bow[i] = 0
        except:
            print(" ")

print(" ")

for bow in documents1:
    for i in range(len(bow)):
        try:
            if (counter[i]+counter1[i]) > 0:
                # bow[i] = bow[i] * (2 * (abs((counter[i]/(counter[i]+counter1[i])) - 0.5)))
                bow[i] = bow[i] * (pow(((counter[i]/(counter[i]+counter1[i])) - 0.5),2)) * 4
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 4)) * 16
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 12)) * 4096
                # x = (counter[i] / (counter[i] + counter1[i]))
                # a = (pow((x - 0.5), 4))
                # bow[i] = bow[i] * a * ((600 * math.exp(x)) / (100 + math.exp(x)))
            else:
                bow[i] = 0
        except:
            print(" ")

# np.stack transforms the list of vectors into a 2D array.
X = np.stack(documents)
Y = np.array(labels)
X1 = np.stack(documents1)
Y1 = np.array(labels1)
X = np.concatenate((X, X1))
Y = np.concatenate((Y, Y1))
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)

documents, labels, counter = read_document("232306-clickbait/clickbait_test.txt", voc, 1)
documents1, labels1, counter1 = read_document("232306-clickbait/non_clickbait_test.txt", voc, 0)

for bow in documents:
    for i in range(len(bow)):
        try:
            if (counter[i]+counter1[i]) > 0:
                # bow[i] = bow[i] * (2 * (abs((counter[i]/(counter[i]+counter1[i])) - 0.5)))
                bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 2)) * 4
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 4)) * 16
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 12)) * 4096
                # x = (counter[i] / (counter[i] + counter1[i]))
                # a = (pow((x - 0.5), 4))
                # bow[i] = bow[i] * a * ((600 * math.exp(x)) / (100 + math.exp(x)))
            else:
                bow[i] = 0
        except:
            print(" ")

for bow in documents1:
    for i in range(len(bow)):
        try:
            if (counter[i]+counter1[i]) > 0:
                # bow[i] = bow[i] * (2 * (abs((counter[i]/(counter[i]+counter1[i])) - 0.5)))
                bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 2)) * 4
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 4)) * 16
                # bow[i] = bow[i] * (pow(((counter[i] / (counter[i] + counter1[i])) - 0.5), 12)) * 4096
                # x = (counter[i] / (counter[i] + counter1[i]))
                # a = (pow((x - 0.5), 4))
                # bow[i] = bow[i] * a * ((600 * math.exp(x))/(100+math.exp(x)))
            else:
                bow[i] = 0
        except:
            print(" ")

# np.stack transforms the list of vectors into a 2D array.
X = np.stack(documents)
Y = np.array(labels)
X1 = np.stack(documents1)
Y1 = np.array(labels1)
X = np.concatenate((X, X1))
Y = np.concatenate((Y, Y1))
# The following line append the labels Y as additional column of the
# array of features so that it can be passed to np.savetxt.
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)

