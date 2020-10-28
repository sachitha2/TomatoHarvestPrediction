import tensorflow
import keras

import matplotlib.pyplot as pyplot
import pickle

from matplotlib import  style

import pandas as pd
import numpy as np
import sklearn

from sklearn import linear_model
from sklearn.utils import shuffle
data = pd.read_csv("AB100.csv", sep=",")


data = data[["Time","Plants", "Harvest" ]]

predict = "Harvest"



X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

# print(X)
best = 0
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.25)

'''

for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
    #
    #
    linear = linear_model.LinearRegression()
    #
    linear.fit(x_train, y_train)
    #
    acc = linear.score(x_test, y_test)
    #
    print("Acuracy ",acc)
    if acc > best:
        best = acc
        print("Best ",best)
        b = i
        with open("harvest"+str(i)+".pickle", "wb") as f:
            pickle.dump(linear,f)


'''
pickle_in = open("harvest39.pickle","rb")

linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)
print("Accuracy - ",acc)
#
# print("Intercept :", linear.intercept_)


# predictions = linear.predict(x_test)
#
# for x in range(len(predictions)):
#     print(predictions[x],x_test[x], y_test[x])



p = linear.predict([[1,200]])
print("Expected -  prediction  : ", p)







style.use("ggplot")
pyplot.scatter(data["Plants"],data[predict])
pyplot.xlabel(predict)
pyplot.ylabel("Plants")
pyplot.show()














