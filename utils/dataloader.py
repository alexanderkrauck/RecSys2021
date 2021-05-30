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
from time import time as ti

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
        random_file_sampling: bool = False,
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
            >=2: Timing information regarding important separate processes
        """

        self.data_directory = data_directory
        self.mode = mode
        self.user_index = pd.read_parquet(user_index_location)
        self.user_index = self.user_index.drop(["following_count", "verified", "following_count", "follower_count", "account_creation"] ,axis=1)
        self.filter_timestamp = filter_timestamp
        self.random_file_sampling = random_file_sampling

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


    def next_file_name__(self):
        if self.random_file_sampling:
            n_remaining = len(self.remaining_files)
            if n_remaining != 1:
                next_idx = np.random.choice(n_remaining)
                return self.remaining_files.pop(next_idx)
        return self.remaining_files.pop(0)

    def next_df__(self):
        if self.batch_size == -1:
            if len(self.remaining_files) == 0:
                raise StopIteration()

            self.current_file_name = self.next_file_name__()

            df = pd.read_csv(
                os.path.join(self.data_directory, self.current_file_name),
                sep='\x01',
                header=None,
                names=self.usecols,
                dtype={k: v for k, v in dtypes_of_features.items() if k in all_features}
            )
        else:
            if self.current_file_name is None:
                self.current_file_name = self.next_file_name__()
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
                total_count += len(try_files[-1])
                if total_count < self.batch_size: #means that this file is done
                    if len(self.remaining_files) == 0:
                        if total_count == 0: #means that all rows of this file have already been returned by a previous iteration
                            raise StopIteration()
                        else:
                            self.current_index += len(try_files[-1])
                            break #break to finish the last batch
                    else:#if there are still files remaining then load next file
                        self.current_index = 0
                        self.current_file_name = self.next_file_name__()
                else:
                    self.current_index += len(try_files[-1])
                    break
            if len(try_files) == 1:
                df = try_files[0]
            else:
                df = pd.concat(try_files, ignore_index=True)
                del try_files
                gc.collect()

        return df

    def __next__(self):
        
        start_delta_t = ti()
        delta_t = ti()
        if self.n_batches_done == self.load_n_batches:
            raise StopIteration()

        df = self.next_df__()

        self.n_batches_done += 1
        if self.verbose >= 2: print(f"Loaded Batch Nr. {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()

        if self.filter_timestamp is not None:
            if self.mode == "train":
                df = df[
                    (df["timestamp"] < self.filter_timestamp) & 
                    ((df["reply"] < self.filter_timestamp) | (df["reply"].isnull())) &
                    ((df["retweet"] < self.filter_timestamp) | (df["retweet"].isnull())) &
                    ((df["like"] < self.filter_timestamp) | (df["like"].isnull())) &
                    ((df["retweet_comment"] < self.filter_timestamp) | (df["retweet_comment"].isnull()))
                ]
            if self.mode == "val":
                df = df[
                    (df["timestamp"] >= self.filter_timestamp) & 
                    ((df["reply"] >= self.filter_timestamp) | (df["reply"].isnull())) &
                    ((df["retweet"] >= self.filter_timestamp) | (df["retweet"].isnull())) &
                    ((df["like"] >= self.filter_timestamp) | (df["like"].isnull())) &
                    ((df["retweet_comment"] >= self.filter_timestamp) | (df["retweet_comment"].isnull()))
                ]
        
        if self.verbose >= 2: print(f"Timestamp Filtered Batch Nr. {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()


        #do prepro 1
        df["medias"] = df["medias"].fillna("")
        df["hashtags"] = df["hashtags"].fillna("")
        df["links"] = df["links"].fillna("")
        df["domains"] = df["domains"].fillna("")
        df["medias"] = df["medias"].fillna("")

        if self.mode != "test":
            df["reply"] = df["reply"].fillna(0).astype(np.uint32)
            df["retweet"] = df["retweet"].fillna(0).astype(np.uint32)
            df["retweet_comment"] = df["retweet_comment"].fillna(0).astype(np.uint32)
            df["like"] = df["like"].fillna(0).astype(np.uint32)

        if self.verbose >= 2: print(f"Did prepro part 1 of {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()


        #do prepro 2


        for key, (cols, fun, dt, iterlevel) in single_column_features.items():
            if iterlevel == 1:
                df[key] = df[cols].apply(fun).astype(dt)
            if iterlevel == 3:
                df[key] = df[cols].apply(fun, axis=1).astype(dt)

        if self.mode != "test":
            for key, (cols, fun, dt, iterlevel) in single_column_targets.items():
                if iterlevel == 1:
                    df[key] = df[cols].apply(fun).astype(dt)
                if iterlevel == 3:
                    df[key] = df[cols].apply(fun, axis=1).astype(dt)


        if self.verbose >= 2: print(f"Did prepro part 2 of {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()
        #do prepro 3 (much faster like this than with apply)
        s = df["a_account_creation"] - df["b_account_creation"]
        sig = np.sign(s)
        df["a_b_creation_delta"] = (np.log(s * sig + 1) * sig).astype(np.float32)

        s = df["timestamp"] - df["a_account_creation"]
        sig = np.sign(s)
        df["a_creation_delta"] = (np.log(s * sig + 1) * sig).astype(np.float32)

        s = df["timestamp"] - df["b_account_creation"]
        sig = np.sign(s)
        df["b_creation_delta"] = (np.log(s * sig + 1) * sig).astype(np.float32)

        if self.verbose >= 2: print(f"Did prepro part 3 of {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()
        #drop not needed cols
        df = df.drop(
            ["bert_base_multilingual_cased_tokens", "hashtags", "medias", "links", "domains", "type", "language", "timestamp"], 
            axis = 1
        )

        #do user_centric prepro
        df["a_user_id_num"] = df["a_user_id"].apply(lambda x: int(x, base=16)%md).astype(np.uint64)
        df["b_user_id_num"] = df["b_user_id"].apply(lambda x: int(x, base=16)%md).astype(np.uint64)


        #this is very RAM costly for short periods of time (+15GB spikes ontop of the normal)
        gc.collect()
        delta_t = ti()
        #df = pd.merge(df, self.user_index, how="left", left_on="a_user_id_num", right_index=True)
        df  = df.join(self.user_index, on="a_user_id_num")
        gc.collect()
        #df = pd.merge(df, self.user_index, how="left", left_on="b_user_id_num", right_index=True, suffixes=("_A", "_B"))  
        df  = df.join(self.user_index, on="b_user_id_num", lsuffix="_A", rsuffix="_B")
        gc.collect()


        if self.verbose >= 2: print(f"Merged Users of {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()


        #Extract TE
        for target in ["reply", "like", "retweet", "retweet_comment"]:
            for as_user in["_a","_b"]:
                prior = (self.user_index[f"n_{target}{as_user}"] / self.user_index[f"n_present{as_user}"]).mean()
                for user in ["_A","_B"]:
                    user_prior = (df[f"n_{target}{as_user}{user}"] / df[f"n_present{as_user}{user}"]).fillna(0)
                    df[f"TE_{target}{as_user}{user}"] = (df[f"n_present{as_user}{user}"] * user_prior + TE_smoothing * prior) / (TE_smoothing + df[f"n_present{as_user}{user}"])
        
        if self.verbose >= 2: print(f"Extracted TE of {self.n_batches_done} in {ti() - delta_t:.2f}")
        delta_t = ti()

        #safety
        df = df.fillna(0)

        #Drop unneccesary cols
        df = df.drop(["a_user_id_num", "b_user_id_num", "a_account_creation", "b_account_creation", "a_user_id"], axis=1)

        if self.verbose >= 1: print(f"Finished Batch Nr. {self.n_batches_done} from file {self.current_file_name} in {ti() - start_delta_t:.2f}s!")
        if self.mode != "test":
            reply, retweet, retweet_comment, like = (list(df["has_reply"]), list(df["has_retweet"]), list(df["has_retweet_comment"]), list(df["has_like"]))
            df = df.drop(["reply", "retweet", "retweet_comment", "like"], axis=1)
            df = df.drop(["has_reply", "has_retweet", "has_retweet_comment", "has_like"], axis=1)
            df = df.drop(["tweet_id", "b_user_id"], axis=1)


            return df, (reply, retweet, retweet_comment, like)


        if self.mode == "test":
            tweet_id, b_user_id = (df["tweet_id"], df["b_user_id"])
            df = df.drop(["tweet_id", "b_user_id"], axis=1)


            return (tweet_id, b_user_id), df









        




