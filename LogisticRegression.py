import numpy as np
import pandas as pd
from main import StandardScaler

class LogisticRegression:
    def __init__(self, X, y, random_state=None):
        self.X = X
        self.y = y
        self.n_classes = np.unique(y)
        self.mapping =  {class_name: idx for idx, class_name in enumerate(self.n_classes)}

        n_features = X.shape[1]
        if random_state is not None:
            np.random.seed(random_state)  # <-- only setting the seed

        if len(self.n_classes) == 2:
            self.w = np.random.randn(n_features) * 0.01
            self.b = 0.0
        else:
            self.w = np.random.randn(n_features, len(self.n_classes)) * 0.01
            self.b = np.zeros(len(self.n_classes))  # Bias also per class        

    def predict_proba(self, X):
        z = self.calc_linear_combination(X)
        if(len(self.n_classes) == 2):
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
                if (len(self.n_classes) == 2):
                    print(self.calc_binary_cross_entropy(z))
                else:
                    print(self.calc_multi_class_cross_entropy(z))

    def calc_gradient_weight(self, z):
        m = self.X.shape[0]
        #binary classification
        if(len(self.n_classes) == 2):
            errors = z - self.y
            return (1 / m) * np.dot(self.X.T, errors)
        else:
            tmp = z.copy()
            counter = 0
            for y, y_ in zip(self.y, tmp):
                y_[self.mapping[y]] = y_[self.mapping[y]] - 1
                tmp[counter] = y_
                counter+=1
            return (1 / m) * np.dot(self.X.T, tmp)

    def calc_gradient_bias(self, z):
        m = self.X.shape[0]
        if len(self.n_classes) == 2:
            sum_ = 0
            for i in range(m):
                sum_ += z[i] - self.y[i]
            return (1 / m) * sum_
        else:
            tmp = z.copy()
            counter = 0
            for y, y_ in zip(self.y, tmp):
                y_[self.mapping[y]] = y_[self.mapping[y]] - 1
                tmp[counter] = y_
                counter+=1
            return (1 / m) * np.sum(tmp, axis=0)

    def calc_binary_cross_entropy(self, y_):
        total_loss = 0
        for i,j in zip(self.y, y_):
            total_loss += -(i * np.log(j) + (1 - i) * np.log(1-j))
        return total_loss / self.y.shape[0]
    
    def calc_multi_class_cross_entropy(self, y_):
        total_loss = 0
        for i,j in zip(self.y, y_):
            total_loss += -np.log(j[self.mapping[i]])
        return total_loss / len(self.y)

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

    train_X = p.drop("diagnosis",axis=1).values
    train_y = p["diagnosis"].values

    l = LogisticRegression(train_X,train_y)
    l.fit()
