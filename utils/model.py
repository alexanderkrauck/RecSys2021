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

class RecSys2021BaseModel():
    """Base class to inherit from for the Recsys2021 challenge"""

    def evaluate_test_set(self, testLoader: torch.utils.DataLoader, output_file: str = "output.csv"):
        """Evaluates the test data inside the testLoader and otputs it appropriatly

        Parameters
        ----------
        testLoader: torch.utils.DataLoader
            The dataloader for the test data. The dataloader is expected to always give the tweet_ids and 
            user_ids (engaging user id = user b) with the batch.
        output_file: str
            The filepath where the results will be written in the appropriate format.
        
        """


        with open(output_file, 'w') as output:
            for tweet_ids, user_ids, batch in testLoader:
                reply_preds, retweet_preds, quote_preds, fav_preds = self.model.forward(batch)

                for tweet_id, user_id, reply_pred, retweet_pred, quote_pred, fav_pred in 
                    zip(tweet_ids, user_ids, reply_preds, retweet_preds, quote_preds, fav_preds)

                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{quote_pred},{fav_pred}\n')
        
        pass
    
    def forward(self):
        """In the forward call the model should process 1 batch"""
        pass


    


from sklearn.metrics import average_precision_score, log_loss

def calculate_ctr(gt):
  positive = len([x for x in gt if x == 1])
  ctr = positive/float(len(gt))
  return ctr

def compute_rce(pred, gt):
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0