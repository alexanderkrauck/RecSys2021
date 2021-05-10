"""
Utility classes for dataloaders for Recsys2021 challenge

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "07-05-2021"

from typing import Union, List, Tuple, Callable, Dict, Iterable
import pandas as pd

class RecSys2021PandasDataLoader():
    """Simple In-Memory Pandas Dataloader
    
    Currently only supports single batch loading
    """

    def __init__(self, data: pd.DataFrame, feature_columns: List = None, mode: str = "test"):
        """

        Parameters
        ----------
        data: pd.DataFrame
            The data which contains the given feature columns and if mode="validation" then also the 4 target columns has_reply, has_retweet, has_retweet_comment, has_like
        feature_columns: List
            A list of the feature column names
        mode: str
            The mode of the DataLoader. Currently supports "test" or "validation". These modes are conform with the "utils.model.RecSys2021BaseModel" class methods.
        """

        self.feature_columns = feature_columns
        self.data = data
        self.mode = mode

    def __iter__(self):
        self.ret = True
        return self

    def __next__(self):
        if self.ret:
            self.ret = False
            if self.mode == "test":
                return (self.data["tweet_id"], self.data["b_user_id"]), self.data[self.feature_columns]
            if self.mode == "validation":
                return self.data[self.feature_columns], (list(self.data["has_reply"]), list(self.data["has_retweet"]), list(self.data["has_retweet_comment"]), list(self.data["has_like"]))
        raise StopIteration()

