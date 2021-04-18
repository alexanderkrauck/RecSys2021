import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class NaiveSet(Dataset):
    """General Dataset to use it as an intermediate structure.
    Key Idea, My more elaborate datasets in getitem method are calling this one, receive preprocessed data
    and label and process them more if needed"""

    def __init__(self, df, stats_file=None):
        self.df = df
        self.stats = stats_file

        # split df and reencode labels
        self.labels = ['has_reply', 'has_retweet', 'has_retweet_comment', 'has_like']
        self.features = [feat for feat in self.df.columns if feat not in self.labels]
        self.encodings = [1, 10, 100, 1000]

        for label, code in zip(self.labels, self.encodings):
            self.df[label] = self.df[label].replace({1: code, 0: 0})

        #to numpy
        self.df['class'] = (self.df['has_reply'] +
                       self.df['has_retweet'] +
                       self.df['has_retweet_comment'] +
                       self.df['has_like'])


    def __len__(self):
        return self.df.shape[0].compute()

    def __getitem__(self,idx):
        row = self.df.loc[idx+1]
        #TODO
        # introduce precalculated features from stats

        #TODO possibly
        # to implement in other dataset classes:
            #convert data and labels to np, and tensors if needed
            #stack features by categories
            #i.e 2D array where 1st row is numerical data on tweet, second categorical, third on user a ...

            #return data and labels (mb smth more for collate_fn or so)

        return self.features[idx], self.labels[idx], idx

    def form_subset(self,tuple,dict=None):
        lst = list(range(tuple[0],tuple[1]))

        #form subset
        features_batch = self.df.loc[lst][self.features].values[:,1:].compute_chunk_sizes()
        labels_batch = self.df.loc[lst]['class'].values.compute_chunk_sizes()

        # dividing training set into chunks, not really used now
        # may be used latter as we come up with more elaborate descisions
        # call rechunk with dict {0:numbatches = n_updates*batches per upd, 1:num of features from our data}
        if dict!=None:
            features_batch.rechunk(dict)
            del dict[0]
            labels_batch.rechunk(dict)
        return (features_batch,labels_batch)