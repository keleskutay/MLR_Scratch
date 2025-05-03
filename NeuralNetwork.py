import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, learning_rate, epochs, hidden_layer_size, activation = 'relu'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation

    def relu_derivative(self, Z1):
        return (Z1 > 0).astype(float)
    
    def tanh_derivative(self, Z1):
        return 1 - np.tanh(Z1) ** 2

    def activation_function(self,Z1):
        if self.activation == 'relu':
            return np.maximum(0, Z1)
        
    def mean_squared_error(self, m, y, y_):
        return np.sum(np.square(y_ - y)) / (2 * m)
    
    def output_gradient_weight(self, y, y_):
        delta2 = y_ - y
        return np.dot(self.A1.T, delta2)
    
    def output_gradient_bias(self, y, y_):
        delta2 = y_ - y
        return np.sum(delta2)
    
    def hidden_layer_error(self, y, y_):
        delta2 = y_ - y
        delta1 = np.dot(delta2,self.W2.T) * self.relu_derivative(self.Z1)
        return delta1
    
    def hidden_gradient_weight(self, X, y, y_):
        delta1 = self.hidden_layer_error(y, y_)
        return np.dot(X.T, delta1)
    
    def hidden_gradient_bias(self, y, y_):
        delta1 = self.hidden_layer_error(y, y_)
        return np.sum(delta1)
    
    def forward(self, X):
        
        self.Z1 =  np.dot(X, self.W1) + self.b1
        self.A1 = self.activation_function(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2

    def backward(self, X, y, Z2):
        self.W1 -= self.learning_rate * self.hidden_gradient_weight(X, y, Z2)
        self.b1 -= self.learning_rate * self.hidden_gradient_bias(y, Z2)

        self.W2 -= self.learning_rate * self.output_gradient_weight(y, Z2)
        self.b2 -= self.learning_rate * self.output_gradient_bias(y, Z2)

    def fit(self, X, y):
        self.W1 = np.random.randn(X.shape[1],self.hidden_layer_size) * 0.001
        self.b1 = np.zeros((1, self.hidden_layer_size))

        self.W2 = np.random.randn(self.hidden_layer_size, 1) * 0.001
        self.b2 = np.zeros((1,))

        for i in range(self.epochs):
            self.forward(X)
            loss = self.mean_squared_error(X.shape[0], y, self.Z2)
            self.backward(X, y, self.Z2)
            
            if i % 500 == 0:
                print(loss)

    def predict(self):
        pass

