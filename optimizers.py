import numpy as np

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params:
            params[key] -= self.lr * grads[f'd{key}']

class Momentum:
    def __init__(self, lr=0.01, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = {k: np.zeros_like(v) for k, v in params.item()}

        for key in params:
            self.velocity[key] = self.beta * self.velocity[key] + (1 - self.beta) * grads[f'd{key}']
            params[key] -= self.lr * self.velocity[key]
