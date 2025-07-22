import numpy as np

class CIFAR10Loader:
    def __init__(self, path='cifar-10-batches-py'):
        self.path = path

    def load_batch(self, fpath):
        import pickle
        with open(fpath, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        X = batch[b'data'].astype(np.float32) / 255.0
        y = np.array(batch[b'labels'], dtype=np.int64)
        return X, y

    def load_data(self):
        import os
        Xs, ys = [], []
        for i in range(1, 6):
            Xi, yi = self.load_batch(os.path.join(self.path, f'data_batch_{i}'))
            Xs.append(Xi); ys.append(yi)
        X_train = np.vstack(Xs); y_train = np.hstack(ys)
        X_test, y_test = self.load_batch(os.path.join(self.path, 'test_batch'))
        return X_train, y_train, X_test, y_test

class MLP:
    def __init__(self, n_input, n_hidden, n_output):
        self.params = self.init_params(n_input, n_hidden, n_output)

    @staticmethod
    def init_params(n_input, n_hidden, n_output):
        return {
            'W1': np.random.randn(n_input, n_hidden) * np.sqrt(2/n_input),
            'b1': np.zeros((1, n_hidden)),
            'W2': np.random.randn(n_hidden, n_output) * np.sqrt(2/n_hidden),
            'b2': np.zeros((1, n_output))
        }

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_grad(Z):
        return (Z > 0).astype(float)

    @staticmethod
    def softmax(Z):
        exp = np.exp(Z - Z.max(axis=1, keepdims=True))
        return exp / exp.sum(axis = 1, keepdims = True)


    def forward(self, X, is_training=True):
        p = self.params
        Z1 = X @ p['W1'] + p['b1']
        A1 = self.relu(Z1)
        Z2 = A1 @ p['W2'] + p['b2']
        A2 = self.softmax(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

    def backward(self, cache, y):
        X, Z1, A1, Z2, A2 = cache
        m = X.shape[0]
        dZ2 = A2.copy(); dZ2[np.arange(m), y] -= 1; dZ2 /= m
        grads = {}
        grads['dW2'] = A1.T @ dZ2
        grads['db2'] = dZ2.sum(axis=0,keepdims=True)
        dA1 = dZ2 @ self.params['W2'].T
        dZ1 = dA1 * self.relu_grad(Z1)
        grads['dW1'] = X.T @ dZ1
        grads['db1'] = dZ1.sum(axis=0, keepdims=True)
        return grads

    def update(self, grads, lr):
        for key in grads:
            self.params[key[1:]] -= lr * grads[key]

    def train(self, X, y, X_val, y_val, lr = 0.01, epochs=50, batch_size=128):
        n = X.shape[0]
        for ep in range(epochs):
            perm = np.random.permutation(n)
            Xs, ys = X[perm], y[perm]
            for i in range(0, n, batch_size):
                xb, yb = Xs[i:i+batch_size], ys[i:i+batch_size]
                A2, cache = self.forward(xb)
                grads = self.backward(cache, yb)
                self.update(grads, lr)


            preds = np.argmax(self.forward(X_val)[0], axis=1)
            acc = np.mean(preds == y_val)
            print(f"Epoch {ep+1}/{epochs}, val_acc = {acc:.4f}")

loader = CIFAR10Loader()
X_train, y_train, X_test, y_test = loader.load_data()

mlp = MLP(n_input=3072, n_hidden=512, n_output=10)
mlp.train(X_train, y_train, X_test, y_test, lr=0.001, epochs=50, batch_size=128)

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_predictions(model, X_test, y_test, n=9):
    plt.figure(figsize=(12, 12))
    indices = np.random.choice(len(X_test), n, replace=False)

    for i, idx in enumerate(indices):
        plt.subplot(3, 3, i + 1)
        img = X_test[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        plt.imshow(img)

        logits, _ = model.forward(X_test[idx:idx + 1] + 1e-8)
        pred_class = classes[np.argmax(logits)]
        true_class = classes[y_test[idx]]

        title_color = 'green' if pred_class == true_class else 'red'
        plt.title(f"True: {true_class}\nPred: {pred_class}", color=title_color)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show(block=True)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

plot_predictions(mlp, X_test, y_test)