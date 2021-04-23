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

#Convention Imports
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Callable, Dict

class RecSys2021BaseModel(ABC):
    """Base class to inherit from for the Recsys2021 challenge"""

    def evaluate_test_set(self, testLoader: torch.utils.data.DataLoader, output_file: str = "output.csv"):
        """Evaluates the test data inside the testLoader and otputs it appropriatly

        Parameters
        ----------
        testLoader: torch.utils.data.DataLoader
            The dataloader for the test data. The dataloader is expected to always give the tweet_ids and 
            user_ids (engaging user id = user b) with the batch.
        output_file: str
            The filepath where the results will be written in the appropriate format.
        
        """


        with open(output_file, 'w') as output:
            for tweet_ids, user_ids, batch in testLoader:
                reply_preds, retweet_preds, retweet_comment_preds, like_preds = self.forward(batch)

                for tweet_id, user_id, reply_pred, retweet_pred, retweet_comment_pred, like_pred in \
                    zip(tweet_ids, user_ids, reply_preds, retweet_preds, retweet_comment_preds, like_preds):

                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{retweet_comment_pred},{like_pred}\n')
        
    
    @abstractmethod
    def forward(self, x: Union[List, torch.Tensor, Tuple, Dict]) -> Tuple:
        """In the forward call the model should process 1 batch
        
        Parameters
        ----------
        x: Union(List, torch.Tensor, Tuple, Dict)
            The input data
        
        Returns
        ----------
        Tuple: (reply_preds, retweet_preds, retweet_comment_preds, like_preds)
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







"""Functions copied from https://recsys-twitter.com/code/snippets"""
from sklearn.metrics import average_precision_score, log_loss
def calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy) * 100.0
  
