from flask import Flask
from flask_restful import Api,Resource
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

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.25)
pickle_in = open("harvest39.pickle","rb")

linear = pickle.load(pickle_in)

acc = linear.score(x_test, y_test)




class HelloWorld(Resource):
    def get(self,name):
        pre = {}
        for i in range(10):
            p = linear.predict([[i+1,name]])
            pre[str(i)] = dict(enumerate(p.flatten(), 1))
                

        return {"data":{
            "accuracy":acc,
            "prediction":pre
        }}
    



app = Flask(__name__)
api = Api(app)


api.add_resource(HelloWorld,"/hello/<int:name>")

if __name__ == "__main__":
    app.run(debug=True)