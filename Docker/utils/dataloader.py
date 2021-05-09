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

    def __init__(self, data: pd.DataFrame, feature_columns: List = None):
        self.feature_columns = feature_columns
        self.data = data

    def __iter__(self):
        self.ret = True
        return self

    def __next__(self):
        if self.ret:
            self.ret = False
            return (self.data["tweet_id"], self.data["b_user_id"]), self.data[self.feature_columns]
        raise StopIteration()

