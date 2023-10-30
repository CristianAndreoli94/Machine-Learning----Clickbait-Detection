import collections
import os
import porter


def remove_punctuation(text):
    """Replace punctuation symbols with spaces."""
    punct = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"  # import string; string.punctuation;
    for p in punct:
        text = text.replace(p, " ")
    return text


def read_document(filename):
    """Read the file and returns a list of words."""
    f = open(filename, encoding="utf8")
    text = f.read()
    f.close()
    words = []
    text = remove_punctuation(text.lower())
    for w in text.split():
        w = porter.stem(w)
        if w.isnumeric():
            num = float(w)
            if num > 0 and num <= 100:
                w = "0_100"
            elif num > 1980 and num <= 2100:
                w = "1980_2100"
            else:
                w = "_NUM"
        words.append(w)
    return words


def write_vocabulary(voc, filename, n):
    """Write the n most frequent words to a file."""
    f = open(filename, "w", encoding="utf8")
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

