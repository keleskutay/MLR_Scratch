import numpy as np
import pandas as pd
from main import StandardScaler

mapping = {
    "M": 0,
    "B" : 1,
    "V": 2,
}

class LogisticRegression:
    def __init__(self, X, y, random_state=None):
        self.X = X
        self.y = y
        self.n_classes = len(np.unique(y))
        n_features = X.shape[1]
        if random_state is not None:
            np.random.seed(random_state)  # <-- only setting the seed

        if self.n_classes == 2:
            self.w = np.random.randn(n_features) * 0.01
            self.b = 0.0
        else:
            self.w = np.random.randn(n_features, self.n_classes) * 0.01
            self.b = np.zeros(self.n_classes)  # Bias also per class        

    def predict_proba(self, X):
        z = self.calc_linear_combination(X)
        if(self.n_classes == 2):
            sigmoid = self.calc_sigmoid(z)
            return sigmoid
        else:
            softmax = self.calc_softmax(z)
            return softmax

    def fit(self):
        self.gradient_descent()

    def gradient_descent(self, learning_rate = 0.0001, epoch = 100000, verbose = True):
        for i in range(epoch):
            z = self.predict_proba(self.X)
            self.w = self.w - learning_rate * self.calc_gradient_weight(z)
            self.b = self.b - learning_rate * self.calc_gradient_bias(z)

            if i % 1000 == 0 and verbose == True:
                print(self.calc_binary_cross_entropy(z))

    def predict(self, X):
        comb = self.calc_linear_combination(X)
        sigmoid = self.calc_sigmoid(comb)
        return sigmoid

    def calc_gradient_weight(self, z):
        m = self.X.shape[0]
        errors = z - self.y
        return (1 / m) * np.dot(self.X.T, errors)

    def calc_gradient_bias(self, z):
        row = self.X.shape[0]
        sum_ = 0
        for i in range(row):
            sum_ += z[i] - self.y[i]
        return (1 / row) * sum_

    def calc_binary_cross_entropy(self, y_):
        total_loss = 0
        for i,j in zip(self.y, y_):
            total_loss += -(i * np.log(j) + (1 - i) * np.log(1-j))
        return total_loss / self.y.shape[0]

    def calc_linear_combination(self, X):
        return np.dot(X, self.w) + self.b
    
    def calc_sigmoid(self, z):
        return 1 / (1 + np.power(np.e, -z))

    def calc_softmax(self, z):
        sf_result = np.zeros_like(z)
        
        for j in range(np.shape(z)[0]):
            exp_row = np.exp(z[j])
            sum_exp_row = np.sum(exp_row)
            sf_result[j] = exp_row / sum_exp_row

        return sf_result





if __name__ == "__main__":
    p = pd.read_csv("Cancer.csv").fillna(0)
    np.random.seed(42)
    scaler = StandardScaler()
    scaler.fit(p)
    p = scaler.transform(p)
    p = p.replace(mapping)

    train_X = p.drop("diagnosis",axis=1).values
    train_y = p["diagnosis"].values

    res = LogisticRegression(train_X,train_y)
    print(res.predict_proba(train_X))
