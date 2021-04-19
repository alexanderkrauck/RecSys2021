import os
import numpy as np
from DataSet import NaiveSet
from NaiveClassifier import ComplementNaiveBayes
from dask import dataframe as dd

def training_naive(how_batch:str, specify_batching='default',printout=1e3):
    """
    :parameter how_batch: str, can obtain 2 values, "set_updates", "default"
    :parameter specify_batching
        if  how_batch = set_updates then specify_batching is a tuple (n_updates:int, n_batches_pro_update:int)
        if  how_batch = default then batches will be formed of size 50k samples and there will be only 1 update
        resulting in only 2 information print outs
    :parameter printout: int
    specifies how often validation is computed, and results are printed out
    default value is 1e3
    """

    #load data to dask, receive a ddf object
    ddf = dd.read_parquet(os.path.abspath("mi2/part.0.parquet"))#forlder with preprocessed df

    #form slitted dataset indicies
    dataset = NaiveSet(ddf)
    size = dataset.__len__()
    train_ids = (0,int(size*(8/10)))
    valid_ids = (int(size*(8/10)),int(size*(9/10)))
    test_ids = (int(size*(9/10)),size+1)

    #perform fitting into model
    if how_batch == 'set_updates':
        n_updates = specify_batching[0] #1e3
        n_batches = specify_batching[1] #pro update
    else:
        batch_size = 50000
        n_batches = int(size/batch_size)
        n_updates = 1

    model = ComplementNaiveBayes()
    classes=[0,1,10,11,100,101,110,111,1000,1001,1010,1011,1100,1101,1110,1111]
    weights = dict(zip(classes,[1]*16))
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
            model.fit(X,y,weights)
            classes_count = model.model.class_count_
            sum_count = sum(classes_count)
            weights = classes_count/sum_count
            weights = dict(zip(classes,weights))
            start = stop
            stop+=int(train_ids[1]/(n_updates*n_batches))


        #TODO (maybe ever never)
        # here the calculations may be delayed and parallelised
        if not update % printout:
            # validate model
            #validation set is huge, validate it even more partially and parallelised,
            X_val, y_val = dataset.form_subset(valid_ids)
            val_loss, val_accuracy, val_f1 = model.validate((X_val, y_val))

            #last part of training subset is efficiently small to fit evaluation
            trn_loss, trn_accuracy, trn_f1 = model.validate((X, y))

            del X_val, y_val #validation set of 2billion items is huge itself, define it only when validate, and then delete

            losses[update - 1][0], losses[update - 1][1] = val_loss, trn_loss
            accuracies[update - 1][0], accuracies[update - 1][1] = val_accuracy, trn_accuracy

            print(f"{'='*10} Update {update} {'='*100}")
            print(f"Losses:\n Training   {trn_loss}\n Validation {val_loss}")
            print(f"\nAccuracies \n Training   {trn_accuracy}\n Validation {val_accuracy}")
            print(f"\nF1 scores: \n Training   {trn_f1}\n Validation {val_f1}")

        update+=1

    #test model
    X_val, y_val = dataset.form_subset(valid_ids)
    X_test, y_test = dataset.form_subset(test_ids)
    val_loss, val_accuracy, val_f1 = model.validate((X_val,y_val))
    test_loss, test_accuracy, test_f1 = model.validate((X_test,y_test))
    print(f"{'='*10} Final result {'='*100}")
    print(f"Losses:\n Validation {val_loss}\n Test       {test_loss}")
    print(f"\nAccuracies: \n Validation {val_accuracy}\n Test       {test_accuracy}")
    print(f"\nF1 scores:  \n Validation {val_f1}\n Test       {test_f1}")

#============================================================================================
#function call
#============================================================================================
training_naive(how_batch='set_updates',specify_batching=(6,2),printout=2)