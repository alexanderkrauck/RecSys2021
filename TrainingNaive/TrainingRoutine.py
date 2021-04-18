import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from DataSet import NaiveSet
from NaiveClassifier import ComplementNaiveBayes
from dask.distributed import Client
from dask import dataframe as dd

#load data to dask, receive a ddf object
ddf = dd.read_parquet(os.path.abspath("mi2"))#forlder with preprocessed df

#form slitted dataset
dataset = NaiveSet(ddf)
size = dataset.__len__[0]
train_ids = (0,int(size*(8/10)))
valid_ids = (int(size*(8/10)),int(size*(9/10)))
test_ids = (int(size*(9/10)),size+1)

X_val, y_val = dataset.form_batch(valid_ids)
X_test, y_test = dataset.form_batch(test_ids)

#perform fitting into model
model = ComplementNaiveBayes()

n_updates = 10 #1e3
n_batches = 1 #pro update
losses = np.zeros((n_updates,2))
accuracies = np.zeros((n_updates,2))

update = 1
start = 0
stop = int(train_ids[1]/(n_updates*n_batches))
while update <= n_updates:
    while stop<=int(train_ids[1]/n_updates)*update:
        model.fit(dataset.form_batch((start,stop)))
        start = stop
        stop+=int(train_ids[1]/(n_updates*n_batches))

    #validate model
    val_loss, val_accuracy = model.validate((X_val,y_val))
    trn_loss, trn_accuracy = model.validate(dataset.form_batch((0,stop)))
    print(f"{'='*10} Update {update} {'='*70}")
    print(f"Losses:\n Training   {trn_loss}\n Validation {val_loss}")
    print(f"Accuracies \n Training   {trn_accuracy}\n Validation {val_accuracy}")
    print(f"Classes count: {[el for el in zip([0,1,2,3,4],model.model.class_count_)]}")

    losses[update-1][0], losses[update-1][1] = val_loss, trn_loss
    accuracies[update-1][0], accuracies[update-1][1] = val_accuracy, trn_accuracy

    update+=1

#test model
val_loss, val_accuracy = model.validate((X_val,y_val))
test_loss, test_accuracy = model.validate((X_test,y_test))
trn_loss, trn_accuracy = model.validate(dataset.form_batch((0,stop)))
print(f"{'='*10} Final result {'='*70}")
print(f"Losses:\n Training   {trn_loss}\n Validation {val_loss}\n Test       {test_loss}")
print(f"Accuracies \n Training   {trn_accuracy}\n Validation {val_accuracy}\n Test       {test_accuracy}")
plt.plot(losses[:,0],range(n_updates))
plt.plot(losses[:,1],range(n_updates))

