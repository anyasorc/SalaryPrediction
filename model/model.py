import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import pickle


ds = pd.read_csv('insert your path to the dataset')
print(ds.shape)
X = ds.iloc[:, 0:1].values
y = ds.iloc[:, 1].values

#split the data into train and test dataset of 80 for train and 20 for test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


LR = LinearRegression()
#trained iwth X_train and y_train data
LR.fit(X_train, y_train)

predicted = LR.predict(X_test)


#save the trained model to disk
pickle.dump(LR, open('model.pkl', 'wb'))

