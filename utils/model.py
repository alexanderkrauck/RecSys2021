"""
Utility classes for models for Recsys2021 challenge

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "08-03-2021"


import os
from os.path import join
from pathlib import Path
import torch
from torch import nn
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, log_loss


#Convention Imports
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Callable, Dict, Iterable

class RecSys2021BaseModel(ABC):
    """Base class to inherit from for the Recsys2021 challenge"""

    def evaluate_test_set(self, testLoader: Iterable, output_file: str = "output.csv"):
        """Evaluates the test data inside the testLoader and otputs it appropriatly

        Parameters
        ----------
        testLoader: Iterable
            The dataloader for the test data. The dataloader is expected to always give the tweet_ids and 
            user_ids (engaging user id = user b) with the batch.
        output_file: str
            The filepath where the results will be written in the appropriate format.
        
        """


        with open(output_file, 'a') as output:
            for (tweet_ids, user_ids), batch in testLoader:
                reply_preds, retweet_preds, retweet_comment_preds, like_preds = self.infer(batch)

                for tweet_id, user_id, reply_pred, retweet_pred, retweet_comment_pred, like_pred in \
                    zip(tweet_ids, user_ids, reply_preds, retweet_preds, retweet_comment_preds, like_preds):

                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{retweet_comment_pred},{like_pred}\n')
        
    def evaluate_validation_set(self, validationLoader: Iterable):
        
        reply_preds = []
        retweet_preds = []
        retweet_comment_preds = []
        like_preds = []

        reply_targets = []
        retweet_targets = []
        retweet_comment_targets = []
        like_targets = []
        
        for batch, (reply_target, retweet_target, retweet_comment_target, like_target) in validationLoader:
            reply_targets.extend(reply_target)
            retweet_targets.extend(retweet_target)
            retweet_comment_targets.extend(retweet_comment_target)
            like_targets.extend(like_target)

            reply_pred, retweet_pred, retweet_comment_pred, like_pred = self.infer(batch)
            reply_preds.extend(reply_pred)
            retweet_preds.extend(retweet_pred)
            retweet_comment_preds.extend(retweet_comment_pred)
            like_preds.extend(like_pred)
        
        reply_rce = compute_rce(reply_preds, reply_targets)
        retweet_rce = compute_rce(retweet_preds, retweet_targets)
        retweet_comment_rce = compute_rce(retweet_comment_preds, retweet_comment_targets)
        like_rce = compute_rce(like_preds, like_targets)

        reply_avg_prec = average_precision_score(reply_targets, reply_preds)
        retweet_avg_prec = average_precision_score(retweet_targets, retweet_preds)
        retweet_comment_avg_prec = average_precision_score(retweet_comment_targets, retweet_comment_preds)
        like_avg_prec = average_precision_score(like_targets, like_preds)

        return (reply_avg_prec, retweet_avg_prec, retweet_comment_avg_prec, like_avg_prec), (reply_rce, retweet_rce, retweet_comment_rce, like_rce)




        

    @abstractmethod
    def infer(self, x: Union[List, torch.Tensor, Tuple, Dict]) -> Tuple[Iterable, Iterable, Iterable, Iterable]:
        """In the infer call the model should process 1 batch
        
        Parameters
        ----------
        x: Union(List, torch.Tensor, Tuple, Dict)
            The input data
        
        Returns
        ----------
        batch_prediction: Tuple[Iterable, Iterable, Iterable, Iterable]
            The biary predictions of (reply_preds, retweet_preds, retweet_comment_preds, like_preds).
        """
        
        raise NotImplementedError("This is abstract!")


class SimpleFFN(torch.nn.Module):
    def __init__(self, n_input_features: int):
        super(SimpleFFN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input_features, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 1),#Just one output!
            nn.Sigmoid()#Because we need binary loss
        )

    def forward(self, x):
        return self.net(x)


class RecSysNeural1(torch.nn.Module, RecSys2021BaseModel):
    def __init__(self, n_input_features: int):
        super(RecSysNeural1, self).__init__()

        self.like_net = SimpleFFN(n_input_features)
        self.reply_net = SimpleFFN(n_input_features)
        self.retweet_comment_net = SimpleFFN(n_input_features)
        self.retweet_net = SimpleFFN(n_input_features)

    def forward(self, x):
        return self.reply_net(x), self.retweet_net(x) ,self.retweet_comment_net(x), self.like_net(x)

    def infer(self, x):
        raise NotImplementedError("Implement this!")


class RecSysXGB1(RecSys2021BaseModel):

    def __init__(self, model_dir: str = None):
        self.clfs_ = {}
        self.targets__ = ['has_reply', 'has_retweet', 'has_retweet_comment', 'has_like']


        if model_dir is not None:
            for filename in os.listdir(model_dir):
                booster = xgb.Booster()
                booster.load_model(join(model_dir,filename))
                self.clfs_[filename] = booster

                

    def train_in_memory(self,
            train_set: pd.DataFrame,
            feature_columns: List,
            xgb_parameters: dict,
            save_dir: str = None
        ):
        """Train in-memory with a pandas train set. 
        
        The train_set is expected to have 4 target columns with names ['has_reply', 'has_retweet', 'has_retweet_comment', 'has_like']

        Parameters
        ----------
        train_set: pd.DataFrame
            Pandas dataframe with features and targets.
        feature_columns: List
            List of column names of the train_set that should be used as training features (X) for the models
        xgb_parameters: dict
            The configuration for the XGB model as specified in https://xgboost.readthedocs.io/en/latest/parameter.html
        save_dir: str
            The directory where models will be stored if given
        """
        
        
        for target in self.targets__:
            dtrain = xgb.DMatrix(train_set[feature_columns], label=train_set[target])
            clf = xgb.train(xgb_parameters, dtrain, 10)

            self.clfs_[target] = clf
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                clf.save_model(join(save_dir, target))
        
            
    def infer(self, x, feature_columns = None):
        if feature_columns is not None:
            x = x[feature_columns]

        x = xgb.DMatrix(x)

        results = []
        for key in self.targets__:
            results.append(self.clfs_[key].predict(x))
        
        return tuple(results)




"""Functions copied from https://recsys-twitter.com/code/snippets"""
def calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy) * 100.0
  
