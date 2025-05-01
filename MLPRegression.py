from NeuralNetwork import NeuralNetwork as nn

class MLPRegression(nn):
    def __init__(self, learning_rate = 0.1, epochs = 100, hidden_layer_size=1):
        super().__init__(learning_rate, epochs, hidden_layer_size)
