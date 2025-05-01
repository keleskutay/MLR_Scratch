import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, learning_rate, epochs, hidden_layer_size):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
    
    def forward(self, X):
        input_size = X.shape[1]
        self.W1 = np.random.randn(input_size,self.hidden_layer_size)
        self.b1 = np.zeros((1, self.hidden_layer_size))

        self.W2 = np.random.randn(self.hidden_layer_size, 1)
        self.b2 = np.zeros(1,)

        return (self.W2, self.b2)

    def backward(self):
        pass

    def fit(self, X, y):
        return self.forward(X)

    def predict(self):
        pass

