import numpy as np
import matplotlib.pyplot as plt


def logreg_inference(X, w, b):
    z = X @ w + b
    p = 1 / (1 + np.exp(-z))
    return p


def cross_entropy(P, Y):
    P = np.clip(P, 0.0001, 0.9999)
    return (-Y * np.log(P) - (1 - Y) * np.log(1-P)).mean()


def logreg_train(X, Y, lambda_, lr=1e-3, steps=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    accs = []
    losses = []
    for step in range(steps):
        P = logreg_inference(X, w, b)
        if step % 1000 == 0:
            loss = cross_entropy(P, Y)
            prediction = (P > 0.5)
            accuracy = (prediction == Y).mean()
            accs.append(accuracy)
            losses.append(loss)
            print(step, loss, accuracy * 100)
        grad_w = ((P - Y) @ X) / m + 2 * lambda_ * w
        grad_b = (P - Y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b, accs, losses


def load_file(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y


X, Y = load_file("train.txt.gz")
print("Loaded", X.shape[0], "feature vectors")


w, b, accs, losses = logreg_train(X, Y, 0, 1e-3, 40000)

# x = np.array([[2, 0, 36, 0, 0, 13.0]])
# p = logreg_inference(x, w, b)
# print("Probability to survive:", p[0] * 100)

print("Weights:", w)
print("Bias:", b)

np.savez("model.npz", w, b)

plt.plot(accs)
plt.figure()
plt.plot(losses)
plt.show()

# Xrnd = X + np.random.randn(X.shape[0], X.shape[1]) / 20
# plt.scatter(Xrnd[:, 0], Xrnd[:, 1], c=Y)
# plt.xlabel("Class")
# plt.ylabel("Sex")
# plt.show()
