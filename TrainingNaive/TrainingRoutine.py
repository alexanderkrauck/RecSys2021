import os
import numpy as np
from matplotlib import pyplot as plt
from DataSet import NaiveSet
from NaiveClassifier import ComplementNaiveBayes
from dask import dataframe as dd

#load data to dask, receive a ddf object
ddf = dd.read_parquet(os.path.abspath("mi2/part.0.parquet"))#forlder with preprocessed df

#form slitted dataset indicies
dataset = NaiveSet(ddf)
size = dataset.__len__()
train_ids = (0,int(size*(8/10)))
valid_ids = (int(size*(8/10)),int(size*(9/10)))
test_ids = (int(size*(9/10)),size+1)

#perform fitting into model
n_updates = 10 #1e3
n_batches = 2 #pro update
model = ComplementNaiveBayes()
classes=[0,1,10,11,100,101,110,111,1000,1001,1010,1011,1100,1101,1110,1111]
losses = np.zeros((n_updates,2))
accuracies = np.zeros((n_updates,2))
update = 1
start = 1
stop = int(train_ids[1]/(n_updates*n_batches))

# could define X y training  after  n_updates and n_batches since we can chuck array
# may be useful in future, see Dataset comments
while update <= n_updates:
    #optimised that we train CNB by giving it data partially, but sequentially
    #didn't find appropriate parallelising routine
    #dask guys also do it sequentially with Increment wrapper
    #wrapper is rather a solution for restricted number models, so I for now stick to mine subsetting
    while stop<=int(train_ids[1]/n_updates)*update:
        X, y = dataset.form_subset((start,stop))
        model.fit(X,y)
        start = stop
        stop+=int(train_ids[1]/(n_updates*n_batches))


    #TODO
    # here the calculations may be delayed and parallelised
    if not update % 3:
        # validate model
        X_val, y_val = dataset.form_subset(valid_ids)
        val_loss, val_accuracy = model.validate((X_val, y_val))
        trn_loss, trn_accuracy = model.validate((X, y))  # last part of training data

        del X_val, y_val #validation set of 2billion items is huge itself, define it only when validate, and then delete

        losses[update - 1][0], losses[update - 1][1] = val_loss, trn_loss
        accuracies[update - 1][0], accuracies[update - 1][1] = val_accuracy, trn_accuracy

        print(f"{'='*10} Update {update} {'='*100}")
        print(f"Losses:\n Training   {trn_loss}\n Validation {val_loss}")
        print(f"\nAccuracies \n Training   {trn_accuracy}\n Validation {val_accuracy}")
        print(f"\nClasses count: \n{[el for el in zip(classes,model.model.class_count_)]}")

    update+=1

#test model
X_val, y_val = dataset.form_subset(valid_ids)
X_test, y_test = dataset.form_subset(test_ids)
val_loss, val_accuracy = model.validate((X_val,y_val))
test_loss, test_accuracy = model.validate((X_test,y_test))
print(f"{'='*10} Final result {'='*100}")
print(f"Losses:\n Validation {val_loss}\n Test       {test_loss}")
print(f"\nAccuracies: \n Validation {val_accuracy}\n Test       {test_accuracy}")