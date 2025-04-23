import numpy as np
import pandas as pd
 
class StandartScaler:
    mapping = {
        "no" : 0,
        "yes" : 1,
        "furnished" : 2,
        "semi-furnished" : 3,
        "unfurnished" : 4
        }
    
    feature_means = {}
    feature_stds = {}
    
    def fit(self, X: pd.DataFrame):
        df = X.select_dtypes(include=np.number)
        for (columnName, columnData) in df.items():
            mean = columnData.mean()
            std = columnData.std()
            self.feature_means[columnName] = mean
            self.feature_stds[columnName] = std

    def transform(self, X: pd.DataFrame):
        df_n = X.select_dtypes(include=np.number)
        df_o = X.select_dtypes(include=[object]).replace(StandartScaler.mapping)

        for (columnName, columnData) in df_n.items():
            for index, row in enumerate(columnData):
                df_n.loc[index, columnName] = (row - self.feature_means[columnName]) / self.feature_stds[columnName]
        
        return pd.concat([df_n, df_o],axis=1)

    def inverse_transform(self, scaled_y):
        y = scaled_y * self.feature_stds["price"] + self.feature_means["price"]
        return y


class LinearRegression:
    def __init__(self, train_X, train_y, b = 0):
        self.train_X = train_X
        self.train_y = train_y
        self.w = np.zeros(np.shape(train_X)[1])
        self.b = b
    
    def fit(self):
        self.gradient_descent()
    
    def predict(self, test_X):
        return np.dot(test_X, self.w) + self.b
    
    def loss_function(self, w, b, X, y):
        predict = 1/2 * np.power((np.dot(X,w) + b) - y, 2)
        tot = 0
        for i in range(predict.shape[0]):
            tot += predict[i]
        return 1 / self.train_X.shape[0] * tot
    
    def gradient_weight(self, w, b):
        m = self.train_X.shape[0]
        predict = np.dot(self.train_X, w) + b
        errors = predict - self.train_y
        return (1 / m) * np.dot(self.train_X.T, errors)

    def gradient_bias(self, w, b):
        row,col = self.train_X.shape[0], self.train_X.shape[1]
        sum_ = 0
        for i in range(row):
            predict = b
            for j in range(col):
                predict += w[j] * self.train_X[i][j]
            sum_ += predict - self.train_y[i]
        return (1 / row) * sum_
    
    def gradient_descent(self, learning_rate = 0.001, epoch = 50000, verbose = True):
        for i in range(epoch):
            self.w = self.w - learning_rate * self.gradient_weight(self.w, self.b)
            self.b = self.b - learning_rate * self.gradient_bias(self.w, self.b)

            if i % 100 == 0 and verbose == True:
                print(self.loss_function(self.w, self.b, self.train_X, self.train_y))


if __name__ == '__main__':
    df = pd.read_csv('./Housing.csv').fillna(0)
    preprocess = StandartScaler()
    preprocess.fit(df)
    processed = preprocess.transform(df)


    train_X = processed.drop("price", axis=1).values
    train_y = processed["price"].values
    
    obj = LinearRegression(train_X, train_y)
    obj.fit()

    test_df = pd.read_csv('./test.csv').fillna(0)
    test_X = preprocess.transform(test_df)
    
    test_X = test_X.drop("price", axis=1).values
    #print(obj.predict(test_X))

    inverse = preprocess.inverse_transform(obj.predict(test_X))
    print(inverse)