import collections
import os


def remove_punctuation(text):
    """Replace punctuation symbols with spaces."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"  # import string; string.punctuation;
    for p in punct:
        text = text.replace(p, " ")
    return text


def read_document(filename):
    """Read the file and returns a list of words."""
    f = open(filename, encoding="utf8")
    stopwords = open("stopwords.txt", encoding="utf8")
    text = f.read()
    stopWordsText = stopwords.read()
    f.close()
    stopwords.close()
    words = []
    text = remove_punctuation(text.lower())
    stopWordsList = stopWordsText.split()
    for w in text.split():
        if len(w) > 2:
            if w not in stopWordsList:
                words.append(w)
    return words


def write_vocabulary(voc, filename, n):
    """Write the n most frequent words to a file."""
    f = open(filename, "w")
    for word, count in voc.most_common(n):
        print(word, file=f)
    f.close()


# The script reads all the documents in the smalltrain directory, uses
# the to form a vocabulary, writes it to the 'vocabulary.txt' file.
# print(read_document("aclImdb/smalltrain/pos/0_9.txt"))

voc = collections.Counter()
voc.update(read_document("232306-clickbait/clickbait_train.txt"))
voc.update(read_document("232306-clickbait/non_clickbait_train.txt"))
write_vocabulary(voc, "vocabulary.txt", 10000)

