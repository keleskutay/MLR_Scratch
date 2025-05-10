import numpy as np
from matplotlib import pyplot as plt

class MNISTLoader:
    def __init__(self, train_x_path = '', train_y_path = ''):
        self.train_x_path = train_x_path
        self.train_y_path = train_y_path

    def get_train_x(self):
        with open(self.train_x_path, 'rb') as train_x:
            train_x.seek(16)
            data = np.frombuffer(train_x.read(), dtype=np.uint8)
            return data.reshape(-1,28,28) / 255.0
    
    def get_train_y(self):
        with open(self.train_y_path, 'rb') as train_y:
            data = np.frombuffer(train_y.read(), dtype=np.uint8)
            return(data[8:])


obj = MNISTLoader('./dataset/train-images.idx3-ubyte', './dataset/train-labels.idx1-ubyte')
X = obj.get_train_x()
y = obj.get_train_y()

