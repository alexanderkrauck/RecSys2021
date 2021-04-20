from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
import numpy as np
import dask


class ComplementNaiveBayes():
    def __init__(self):
        self.model = ComplementNB()

    def fit(self,X,y,dict):
        """Fuction which takes a df or ddf and returns model
        Use partial fit to provide subset training
        :parameter X nd array of features
        :parameter y 1d array of labels
        :parameter weights is a dict, each key-value pair has a following structure (class:weight)"""
        weights = dask.array.zeros_like(y)
        for key in dict:
            weights+=np.where(y == key, dict[key], 0)
        self.model.partial_fit(X, y,classes=[0,1,10,11,
                                             100,101,110,111,
                                             1000,1001,1010,1011,1100,1101,1110,1111],
                                    sample_weight=weights)

    def validate(self,batch):
        features, labels = batch
        prediction = self.model.predict(features)

        accuracy = acc(labels, prediction)
        f_one = f1(labels, prediction, average='weighted')

        #divide by max label to scale loss
        labels = labels/1111
        prediction = prediction/1111
        loss = mse(labels,prediction)
        return (loss, accuracy, f_one)

    def validate_parallel(self,batch):
        """
        :parameter batch, ddfs of labels and features
        is merged to a common ddf, then set of it is formed, then substes again formed,
        each subset is fed into validation, with delayed computation
        individual computations are averaged
        :return: (loss, accuracy, f_one)
        """