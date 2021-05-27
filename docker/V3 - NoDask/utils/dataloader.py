"""
Utility classes for dataloaders for Recsys2021 challenge

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "07-05-2021"

from typing import Union, List, Tuple, Callable, Dict, Iterable
import pandas as pd
import numpy as np
import os
import gc

from .constants import all_columns, dtypes_of_features, all_features
from .features import single_column_features, single_column_targets

md = 2**64
TE_smoothing = 20

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


class RecSys2021TSVDataLoader():
    def __init__(
        self, 
        data_directory: str, 
        user_index_location: str, 
        mode: str = "train", 
        batch_size: int = -1, 
        load_n_batches: int = 1, 
        filter_timestamp: int = None,
        verbose:int = 0):
        """

        Parameters
        ----------
        data_directory: str
            The directory of the data where the data has csv/tsv format.
        user_index_location: str
            The file name of the user index to use for enhancing the data
        mode: str
            Mode of the Dataloader. Can be "train", "val" and "test".
            For "train" and "val" labeles are expected and for "test" it does not expect labels but it also returns tweet_id and b_user_id.
        batch_size: int
            The number of rows to be returned in one batch. 
            If the batch size is larger than the number of rows avaiable in all the data then a batch with size smaller than the given batch size will be returned.
            If batch_size = -1, then exactly 1 part file will be used.
        load_n_batches: int
            The number of batches after which to stop iteration. If -1 then all data will be iterated.
        filter_timestamp: int
            If supplied for "train" mode it only returns data which happens before the timestam and for "val" it only returns data after the timestamp.
            "test" data is unaffected.
        verbose: int
            Level of verboseness.
            <=0: No prints
            >=1: Information regarding each batch
        """

        self.data_directory = data_directory
        self.mode = mode
        self.user_index = pd.read_parquet(user_index_location)
        self.user_index = self.user_index.drop(["following_count", "verified", "following_count", "follower_count", "account_creation"] ,axis=1)
        self.filter_timestamp = filter_timestamp

        self.batch_size = batch_size
        self.load_n_batches = load_n_batches
        self.usecols = all_columns if mode != "test" else all_features
        self.verbose = verbose
        
    def __iter__(self):
        self.remaining_files = os.listdir(self.data_directory)

        self.current_file_name = None
        self.n_batches_done = 0
        self.current_index = 0
        return self

    def __next__(self):
        
        if self.n_batches_done == self.load_n_batches:
            raise StopIteration()

        if self.batch_size == -1:
            if len(self.remaining_files) == 0:
                raise StopIteration()

            next_file_idx = np.random.randint(0, len(self.remaining_files))
            self.current_file_name = self.remaining_files.pop(next_file_idx)

            self.current_file = pd.read_csv(
                os.path.join(self.data_directory, self.current_file_name),
                sep='\x01',
                header=None,
                names=self.usecols,
                dtype={k: v for k, v in dtypes_of_features.items() if k in all_features}
            )
        else:
            try_files = []
            total_count = 0
            while True:
                try_files.append(pd.read_csv(
                    os.path.join(self.data_directory, self.current_file_name),
                    sep='\x01',
                    header=None,
                    names=self.usecols,
                    dtype={k: v for k, v in dtypes_of_features.items() if k in all_features},
                    skiprows=self.current_index,
                    nrows = self.batch_size - total_count
                ))
                total_count+=len(try_files[-1])
                if total_count < self.batch_size:
                    if len(self.remaining_files) == 0:
                        if total_count == 0:
                            raise StopIteration()
                        else:
                            break
                    self.current_index = 0
                    next_file_idx = np.random.randint(0, len(self.remaining_files))
                    self.current_file_name = self.remaining_files.pop(next_file_idx)
                else:
                    self.current_index += self.batch_size
                    break
            if len(try_files == 1):
                self.current_file = try_files[0]
            else:
                self.current_file = pd.concat(try_files)

        self.n_batches_done += 1 

        if self.filter_timestamp is not None:
            if self.mode == "train":
                self.current_file = self.current_file[
                    (self.current_file["timestamp"] < self.filter_timestamp) & 
                    ((self.current_file["reply"] < self.filter_timestamp) | (self.current_file["reply"].isnull())) &
                    ((self.current_file["retweet"] < self.filter_timestamp) | (self.current_file["retweet"].isnull())) &
                    ((self.current_file["like"] < self.filter_timestamp) | (self.current_file["like"].isnull())) &
                    ((self.current_file["retweet_comment"] < self.filter_timestamp) | (self.current_file["retweet_comment"].isnull()))
                ]
            if self.mode == "val":
                self.current_file = self.current_file[
                    (self.current_file["timestamp"] >= self.filter_timestamp) & 
                    ((self.current_file["reply"] >= self.filter_timestamp) | (self.current_file["reply"].isnull())) &
                    ((self.current_file["retweet"] >= self.filter_timestamp) | (self.current_file["retweet"].isnull())) &
                    ((self.current_file["like"] >= self.filter_timestamp) | (self.current_file["like"].isnull())) &
                    ((self.current_file["retweet_comment"] >= self.filter_timestamp) | (self.current_file["retweet_comment"].isnull()))
                ]


        #do prepro 1
        self.current_file["medias"] = self.current_file["medias"].fillna("")
        self.current_file["hashtags"] = self.current_file["hashtags"].fillna("")
        self.current_file["links"] = self.current_file["links"].fillna("")
        self.current_file["domains"] = self.current_file["domains"].fillna("")
        self.current_file["medias"] = self.current_file["medias"].fillna("")

        if self.mode != "test":
            self.current_file["reply"] = self.current_file["reply"].fillna(0).astype(np.uint32)
            self.current_file["retweet"] = self.current_file["retweet"].fillna(0).astype(np.uint32)
            self.current_file["retweet_comment"] = self.current_file["retweet_comment"].fillna(0).astype(np.uint32)
            self.current_file["like"] = self.current_file["like"].fillna(0).astype(np.uint32)


        #do prepro 2
        for key, (cols, fun, dt, iterlevel) in single_column_features.items():
            if iterlevel == 1:
                self.current_file[key] = self.current_file[cols].apply(fun).astype(dt)
            if iterlevel == 3:
                self.current_file[key] = self.current_file[cols].apply(fun, axis=1).astype(dt)

        if self.mode != "test":
            for key, (cols, fun, dt, iterlevel) in single_column_targets.items():
                if iterlevel == 1:
                    self.current_file[key] = self.current_file[cols].apply(fun).astype(dt)
                if iterlevel == 3:
                    self.current_file[key] = self.current_file[cols].apply(fun, axis=1).astype(dt)
        

        #drop not needed cols
        self.current_file = self.current_file.drop(
            ["bert_base_multilingual_cased_tokens", "hashtags", "medias", "links", "domains", "type", "language", "timestamp"], 
            axis = 1
        )

        #do user_centric prepro
        self.current_file["a_user_id_num"] = self.current_file["a_user_id"].apply(lambda x: int(x, base=16)%md).astype(np.uint64)
        self.current_file["b_user_id_num"] = self.current_file["b_user_id"].apply(lambda x: int(x, base=16)%md).astype(np.uint64)

        #this is very RAM costly for short periods of time (+15GB spikes ontop of the normal)
        gc.collect()
        self.current_file = pd.merge(self.current_file, self.user_index, how="left", left_on="a_user_id_num", right_index=True)
        gc.collect()
        self.current_file = pd.merge(self.current_file, self.user_index, how="left", left_on="b_user_id_num", right_index=True, suffixes=("_A", "_B"))  
        gc.collect()


        #Extract TE
        for target in ["reply", "like", "retweet", "retweet_comment"]:
            for as_user in["_a","_b"]:
                prior = (self.user_index[f"n_{target}{as_user}"] / self.user_index[f"n_present{as_user}"]).mean()
                for user in ["_A","_B"]:
                    user_prior = (self.current_file[f"n_{target}{as_user}{user}"] / self.current_file[f"n_present{as_user}{user}"]).fillna(0)
                    self.current_file[f"TE_{target}{as_user}{user}"] = (self.current_file[f"n_present{as_user}{user}"] * user_prior + TE_smoothing * prior) / (TE_smoothing + self.current_file[f"n_present{as_user}{user}"])
        
        #safety
        self.current_file.fillna(0)

        #Drop unneccesary cols
        self.current_file = self.current_file.drop(["a_user_id_num", "b_user_id_num", "a_account_creation", "b_account_creation", "a_user_id"], axis=1)

        if self.verbose >= 1: print(f"Finished Batch Nr. {self.n_batches_done} from file {self.current_file_name}!")
        if self.mode != "test":
            reply, retweet, retweet_comment, like = (list(self.current_file["has_reply"]), list(self.current_file["has_retweet"]), list(self.current_file["has_retweet_comment"]), list(self.current_file["has_like"]))
            self.current_file = self.current_file.drop(["reply", "retweet", "retweet_comment", "like"], axis=1)
            self.current_file = self.current_file.drop(["has_reply", "has_retweet", "has_retweet_comment", "has_like"], axis=1)
            self.current_file = self.current_file.drop(["tweet_id", "b_user_id"], axis=1)


            return self.current_file, (reply, retweet, retweet_comment, like)


        if self.mode == "test":
            tweet_id, b_user_id = (self.current_file["tweet_id"], self.current_file["b_user_id"])
            self.current_file = self.current_file.drop(["tweet_id", "b_user_id"], axis=1)


            return (tweet_id, b_user_id), self.current_file









        




