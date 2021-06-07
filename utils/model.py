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
import torch.nn.functional as F
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, log_loss


from .constants import user_group_weights, like_weights, reply_weights, retweet_comment_weights, retweet_weights

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
            for batch, quantile, (tweet_ids, user_ids) in testLoader:
                reply_preds, retweet_preds, retweet_comment_preds, like_preds = self.infer(batch)

                for tweet_id, user_id, reply_pred, retweet_pred, retweet_comment_pred, like_pred in \
                    zip(tweet_ids, user_ids, reply_preds, retweet_preds, retweet_comment_preds, like_preds):

                    output.write(f'{tweet_id},{user_id},{reply_pred},{retweet_pred},{retweet_comment_pred},{like_pred}\n')
        

    def evaluate_validation_set(self, validationLoader: Iterable):
        """Calculates the official RecSys metrics for a dataset

        Parameters
        ----------
        validationLoader: Iterable
            The dataloader for the validation data. The dataloader is expected to always return a tuple of the batch,
            plus the labels of the batch in the order: batch, (reply_target, retweet_target, retweet_comment_target, like_target)
        
        Returns
        ----------
        Average Precision Score and RCE Score of each target in the same order as the input targets.
        I.e. (reply_avg_prec, retweet_avg_prec, retweet_comment_avg_prec, like_avg_prec), (reply_rce, retweet_rce, retweet_comment_rce, like_rce)
        """
        
        reply_preds = []
        retweet_preds = []
        retweet_comment_preds = []
        like_preds = []

        reply_targets = []
        retweet_targets = []
        retweet_comment_targets = []
        like_targets = []

        quantiles = []
        
        for batch, quantile, (reply_target, retweet_target, retweet_comment_target, like_target) in validationLoader:
            reply_targets.extend(reply_target)
            retweet_targets.extend(retweet_target)
            retweet_comment_targets.extend(retweet_comment_target)
            like_targets.extend(like_target)

            reply_pred, retweet_pred, retweet_comment_pred, like_pred = self.infer(batch, quantile)
            reply_preds.extend(reply_pred)
            retweet_preds.extend(retweet_pred)
            retweet_comment_preds.extend(retweet_comment_pred)
            like_preds.extend(like_pred)

            quantiles.extend(quantile)
        
        reply_preds = np.array(reply_preds)
        retweet_preds = np.array(retweet_preds)
        retweet_comment_preds = np.array(retweet_comment_preds)
        like_preds = np.array(like_preds)

        reply_targets = np.array(reply_targets)
        retweet_targets = np.array(retweet_targets)
        retweet_comment_targets = np.array(retweet_comment_targets)
        like_targets = np.array(like_targets)

        quantiles = np.array(quantiles)


        result_groups={}

        reply_rces = []
        retweet_rces = []
        retweet_comment_rces = []
        like_rces = []

        reply_avg_precs = []
        retweet_avg_precs = []
        retweet_comment_avg_precs = []
        like_avg_precs = []

        for q in range(1, 6):
            reply_rces.append(compute_rce(reply_preds[quantiles == q], reply_targets[quantiles == q]))
            retweet_rces.append(compute_rce(retweet_preds[quantiles == q], retweet_targets[quantiles == q]))
            retweet_comment_rces.append(compute_rce(retweet_comment_preds[quantiles == q], retweet_comment_targets[quantiles == q]))
            like_rces.append(compute_rce(like_preds[quantiles == q], like_targets[quantiles == q]))

            reply_avg_precs.append(average_precision_score(reply_targets[quantiles == q], reply_preds[quantiles == q]))
            retweet_avg_precs.append(average_precision_score(retweet_targets[quantiles == q], retweet_preds[quantiles == q]))
            retweet_comment_avg_precs.append(average_precision_score(retweet_comment_targets[quantiles == q], retweet_comment_preds[quantiles == q]))
            like_avg_precs.append(average_precision_score(like_targets[quantiles == q], like_preds[quantiles == q]))

            result_groups[f"Q{q}_reply_rce"] = reply_rces[-1]
            result_groups[f"Q{q}_retweet_rce"] = retweet_rces[-1]
            result_groups[f"Q{q}_retweet_comment_rce"] = retweet_comment_rces[-1]
            result_groups[f"Q{q}_like_rce"] = like_rces[-1]

            result_groups[f"Q{q}_reply_avg_prec"] = reply_avg_precs[-1]
            result_groups[f"Q{q}_retweet_avg_prec"] = retweet_avg_precs[-1]
            result_groups[f"Q{q}_retweet_comment_avg_prec"] = retweet_comment_avg_precs[-1]
            result_groups[f"Q{q}_like_avg_prec"] = like_avg_precs[-1]

        result_groups[f"TOTAL_reply_rce"] = np.mean(reply_rces)
        result_groups[f"TOTAL_retweet_rce"] = np.mean(retweet_rces)
        result_groups[f"TOTAL_retweet_comment_rce"] = np.mean(retweet_comment_rces)
        result_groups[f"TOTAL_like_rce"] = np.mean(like_rces)

        result_groups[f"TOTAL_reply_avg_prec"] = np.mean(reply_avg_precs)
        result_groups[f"TOTAL_retweet_avg_prec"] = np.mean(retweet_avg_precs)
        result_groups[f"TOTAL_retweet_comment_avg_prec"] = np.mean(retweet_comment_avg_precs)
        result_groups[f"TOTAL_like_avg_prec"] = np.mean(like_avg_precs)
        
        
        return result_groups




        

    @abstractmethod
    def infer(self, x: Union[List, torch.Tensor, Tuple, Dict], quantile: List[int]) -> Tuple[Iterable, Iterable, Iterable, Iterable]:
        """In the infer call the model should process 1 batch
        
        Parameters
        ----------
        x: Union(List, torch.Tensor, Tuple, Dict)
            The input data
        quantile: List[int]
            The quantiles of the input data
        
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
    def __init__(self, model_dir, n_input_features: int, device="cpu"):
        super(RecSysNeural1, self).__init__()

        self.like_net = SimpleFFN(n_input_features)
        self.reply_net = SimpleFFN(n_input_features)
        self.retweet_comment_net = SimpleFFN(n_input_features)
        self.retweet_net = SimpleFFN(n_input_features)

        self.device = device

    def forward(self, x):
        x = torch.tensor(x.to_numpy().astype(np.float32), dtype=torch.float, device=self.device)


        return self.reply_net(x).squeeze(), self.retweet_net(x).squeeze() ,self.retweet_comment_net(x).squeeze(), self.like_net(x).squeeze()

    def fit(self, dl, n_epochs, lr = 1e-3):
        optimizer = torch.optim.Adam(self.parameters(), lr = lr)

        for ep in range(1, n_epochs+1):
            for x, quantile, (reply, retweet, retweet_comment, like) in dl:

                reply_wei = torch.tensor([reply_weights[it] for it in reply], device=self.device, dtype=torch.float)
                like_wei = torch.tensor([like_weights[it] for it in like], device=self.device, dtype=torch.float)
                retweet_comment_wei = torch.tensor([retweet_comment_weights[it] for it in retweet_comment], device=self.device, dtype=torch.float)
                retweet_wei = torch.tensor([retweet_weights[it] for it in retweet], device=self.device, dtype=torch.float)

                reply = torch.tensor(reply, device=self.device, dtype=torch.float)
                retweet = torch.tensor(retweet, device=self.device, dtype=torch.float)
                retweet_comment = torch.tensor(retweet_comment, device=self.device, dtype=torch.float)
                like = torch.tensor(like, device=self.device, dtype=torch.float)

                quantile_wei = torch.tensor([user_group_weights[it] for it in quantile], device=self.device, dtype=torch.float)

                reply_pred, retweet_pred ,retweet_comment_pred, like_pred = self.forward(x)

                reply_loss = F.binary_cross_entropy(reply_pred, reply, reduction="none") * quantile_wei * reply_wei
                retweet_loss = F.binary_cross_entropy(retweet_pred, retweet, reduction="none") * quantile_wei * retweet_wei
                retweet_comment_loss = F.binary_cross_entropy(retweet_comment_pred, retweet_comment, reduction="none") * quantile_wei * retweet_comment_wei
                like_loss = F.binary_cross_entropy(like_pred, like, reduction="none") * quantile_wei * like_wei

                loss = reply_loss.mean() + retweet_comment_loss.mean() + retweet_loss.mean() + like_loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    @torch.no_grad()
    def infer(self, x, quantile = None):
        preds = self.forward(x)
        return (list(preds[0].detach().cpu().numpy()), list(preds[1].detach().cpu().numpy()), list(preds[2].detach().cpu().numpy()), list(preds[3].detach().cpu().numpy()))

    def to(self, device):
        ret = super(RecSysNeural1, self).to(device)
        self.device = device
        return ret

class RecSysXGB1(RecSys2021BaseModel):

    def __init__(self, model_dir):
        self.clfs_ = {}
        self.targets__ = ['has_reply', 'has_retweet', 'has_retweet_comment', 'has_like']

        self.model_dir = model_dir
        self.load()

                
    def save(self):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        for clf_target in self.clfs_:
            clf = self.clfs_[clf_target]
            clf.save_model(join(self.model_dir, clf_target))
    
    def load(self):
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(self.model_dir):
            if filename in self.targets__:
                booster = xgb.Booster()
                booster.load_model(join(self.model_dir, filename))
                self.clfs_[filename] = booster



    def train_in_memory(self,
            train_set: pd.DataFrame,
            quantiles: list,
            targets: tuple,
            xgb_parameters: dict,
            num_boost_rounds: int = 100,
            feature_columns: List = None,
            resume_if_exists: bool = False,
            save_to_model_dir: bool = True,
            verbose: int = 0
        ):
        """Train in-memory with a pandas train set. 
        

        Parameters
        ----------
        train_set: pd.DataFrame
            Pandas dataframe with features to be used.
        targets: tuple
            A tuple with 4 lists for the targets in the order: 'has_reply', 'has_retweet', 'has_retweet_comment', 'has_like'
        xgb_parameters: dict
            The configuration for the XGB model as specified in https://xgboost.readthedocs.io/en/latest/parameter.html
        feature_columns: List
            List of column names of the train_set that should be used as training features (X) for the models.
            If None then all features are used
        """
        
        if feature_columns is None:
            feature_columns = [c for c in train_set.columns if c not in self.targets__]
            
        for target_name, target in zip(self.targets__, targets):
            if verbose >= 1: print(f"Now training {target_name} clf.")
            dtrain = xgb.DMatrix(train_set[feature_columns], label=target)
            if target_name in self.clfs_ and resume_if_exists:
                clf = xgb.train(xgb_parameters, dtrain, num_boost_rounds, xgb_model=self.clfs_[target_name])
            else:
                clf = xgb.train(xgb_parameters, dtrain, num_boost_rounds)

            self.clfs_[target_name] = clf
            if verbose >= 1: print(f"Finished training {target_name} clf.")
        if save_to_model_dir:
            self.save()
        if verbose >= 1: print(f"Finished training all clfs.")


    def fit(self,
            train_loader: Iterable,
            xgb_parameters: dict,
            boost_rounds_per_iteration: int = 10,
            verbose: int = 0,
            n_epochs: int = 1
        ):
        """Train in-memory with a pandas train set. 
        

        Parameters
        ----------
        train_loader: Iterable
            The dataloader for the train data.
        xgb_parameters: dict
            The configuration for the XGB model as specified in https://xgboost.readthedocs.io/en/latest/parameter.html
        verbose: int
            Level of verboseness.
        n_epochs: int
            Number of times the loader will be iterated.
        """
        

        for ep in range(1, n_epochs+1):
            for df, quantile, target_tuple in train_loader:
                self.train_in_memory(
                    df, 
                    quantile, 
                    target_tuple, 
                    xgb_parameters, 
                    boost_rounds_per_iteration, 
                    resume_if_exists=True, 
                    save_to_model_dir=False
                )
            if verbose >= 1: print(f"Finished {ep} epochs.")
        self.save()

            
        
            
    def infer(self, x, quantile = None):

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
    pred = (pred - 0.5)*0.999 + 0.5#safety, to allow 0 or 1 values
    cross_entropy = log_loss(gt, pred)
    data_ctr = calculate_ctr(gt)
    strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
    return (1.0 - cross_entropy/strawman_cross_entropy) * 100.0
  
