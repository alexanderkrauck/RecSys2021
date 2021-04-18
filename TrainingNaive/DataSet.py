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
        self.encodings = [1, 2, 3, 4]

        for label, code in zip(self.labels, self.encodings):
            self.df[label].replace({1: code, 0: 0}, inplace=True)

        #to numpy
        self.df['class'] = (self.df['has_reply'] +
                       self.df['has_retweet'] +
                       self.df['has_retweet_comment'] +
                       self.df['has_like'])
        self.labels = self.df['class'].values
        self.features = self.df[self.features].values


    def __len__(self):
        return self.df.shape

    def __getitem__(self,idx):
        row = self.df.loc[idx+1]
        #TODO
        # introduce precalculated features from stats

        #TODO possibly
        # to implement in othe dataset classes:
            #convert data and labels to np, and tensors if needed
            #stack features by categories
            #i.e 2D array where 1st row is numerical data on tweet, second categorical, third on user a ...

            #return data and labels (mb smth more for collate_fn or so)

        return self.features[idx], self.labels[idx], idx

    def form_batch(self,tuple):
        features_batch = self.features[tuple[0]:tuple[1]]
        labels_batch = self.labels[tuple[0]:tuple[1]]
        return (features_batch,labels_batch)