import numpy as np
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import  mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc

class ComplementNaiveBayes():
    def __init__(self):
        self.model = ComplementNB()

    def fit(self,X,y):
        """Fuction which takes a df or ddf and returns model
        Use partial fit to provide batch training"""
        self.model.partial_fit(X, y,classes=[0,1,10,11,
                                             100,101,110,111,
                                             1000,1001,1010,1011,1100,1101,1110,1111])

    def validate(self,batch):
        features, labels = batch
        prediction = self.model.predict(features)

        accuracy = acc(labels, prediction)

        #divide by max label to scale loss
        labels = labels/1111
        prediction = prediction/1111
        loss = mse(labels,prediction)
        return (loss,accuracy)

