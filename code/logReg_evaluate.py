import numpy as np
# import matplotlib.pyplot as plt

# Load the parameters of the trained model.
data = np.load("model.npz")
w = data["arr_0"]
b = data["arr_1"]
print(w, b)


def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


def logreg_inference(X, w, b):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


X, Y = load_file("test.txt.gz")
print("Loaded", X.shape[0], "feature vectors")


P = logreg_inference(X, w, b)
predictions = (P > 0.5)
accuracy = (Y == predictions).mean()
print("Test accuracy:", accuracy * 100)
