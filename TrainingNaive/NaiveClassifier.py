import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import  mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc

class ComplementNaiveBayes():
    def __init__(self):
        self.model = ComplementNB()

    def fit(self,batch:tuple):
        """Fuction which takes a df or ddf and returns model
        Use partial fit to provide batch training"""
        features, labels = batch
        self.model.partial_fit(features, labels,classes=[0,1,2,3,4])

    def validate(self,batch):
        features, labels = batch
        prediction = self.model.predict(features)
        loss = mse(labels,prediction)
        accuracy = acc(labels,prediction)
        return (loss,accuracy)

