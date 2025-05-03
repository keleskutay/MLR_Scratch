import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MLPRegression import MLPRegression

p = pd.read_csv("china_gdp.csv").fillna(0)
train_X = p.drop(["Value"],axis=1).values
train_y = p["Value"].values.reshape(-1,1)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)

a = MLPRegression(hidden_layer_size=450, learning_rate=0.00001, epochs=5000000)
print(a.fit(train_X, train_y))

