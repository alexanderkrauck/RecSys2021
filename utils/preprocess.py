"""
Utility functions for preprocessing files from the RecSys2020 Servers

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "08-03-2021"


import os
import gc
from os.path import join
from pathlib import Path
import itertools
from time import time as t

import numpy as np

import pandas as pd
import dask
import dask.dataframe as dd


__all_features = ["bert_base_multilingual_cased_tokens",
                "hashtags",
                "tweet_id",
                "medias",
                "links",
                "domains",
                "type",
                "language",
                "timestamp",
                "a_user_id",
                "a_follower_count",
                "a_following_count",
                "a_is_verified",
                "a_account_creation",
                "b_user_id",
                "b_follower_count",
                "b_following_count",
                "b_is_verified",
                "b_account_creation",
                "a_follows_b"] #as far as I know from the forum (b always follows a in this dataset according to the forum)

__all_labels = ["reply",
              "retweet",
              "retweet_comment",
              "like"]

__all_columns = __all_features + __all_labels

__type_mapping = {"Retweet": 0, "Quote":1, "Reply":2, "Toplevel":3}

__media_type_mapping = {"Photo":0, "Video":1, "Gif":2}

#Helper Functions
def _partition_indexing(df, partition_info=None):
    df["tweet_id"] = 1
    df["tweet_id"] = df["tweet_id"].cumsum()
    #This is not a stable choice but will work well with 64MB data splits
    df["tweet_id"] = df["tweet_id"] + partition_info["number"] * 200000 
    return df["tweet_id"]

def _map_list_column(df, mapping, column_name):
    new_column = df[column_name].apply(lambda x: [mapping[str(item)] for item in x])
    return new_column

def _column_length(df, column_name):
    column = df[column_name]
    return column.apply(lambda x: len(str(x).split("\t")))


def initial_preprocess_data(
        csv_path: str,
        output_dir: str,
        convert_tweet_id_to_number: bool,
        seperate_bert: bool,
        fully_drop_bert: bool,
        drop_tweet_id: bool,
        add_bert_length_as_feature: bool = True,
        blocksize: str = "64MB",
        verbose: int = 1
        #TODO: REPLACE USER IDS BY NUMBERS, COLLECT FEATURES THAT REQUIRE STATISTICS OVER MULTIPLE PART FILES
    ):
    """A helper function for the initial heavy processing of the data (will take a bit)
    
    It is recommended to (at least in the the exploration phase) only include a subset of the data,
    since it will take very long otherwise.

    Parameters
    ----------
    csv_path : str
        The description path of the data. ('*' is allowed, we load multiple files at the same time)
    output_dir : str
        The output directory, where the parquet-chunkfiles will be stored
    convert_tweet_id_to_number : bool
        If true the tweet id will be replaced by a unique number, however there will be gaps and this is also unstable
        TODO: make it stable
    seperate_bert : bool
        If true, the BERT column will be seperated into another file which can be joined again over the "tweet_id" field
    fully_drop_bert : bool
        Overrides "seperate_bert". If true, then the BERT column will be dropped and it will not be kept at all.
    drop_tweet_id : bool
        If true the tweet id will be dropped. Also this means that we can not locate the BERT encodings anymore if they are seperated.
        It makes no sense to have "convert_tweet_id_to_number" enabled if this is true.
    add_bert_length_as_feature: bool
        If true then the number of tokens in the bert column will be stored as the new column "bert_length".
    blocksize : str
        The blocksize in which the data of in the path will be split. If this is chosen to high, the memory will become full.
    verbose : int
        Decides level of verboseness. (Max 2, Min 0)
    """
    st = t()
    nobert_dir = join(output_dir,"nobert")
    Path(nobert_dir).mkdir(parents=True, exist_ok=True)


    ddf = dd.read_csv(csv_path, sep='\x01', header=None, names=__all_columns, blocksize=blocksize)
    if verbose > 0: print(f"The data is split into {ddf.npartitions} partitions of size {blocksize} each.")


    ddf['medias'] = ddf['medias'].fillna("")
    ddf['medias'] = ddf['medias'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)
    medias_task = ddf.map_partitions(lambda x: (list(set([a for b in x.medias.tolist() for a in b]))), meta=list)
    if verbose > 0: print("Now computing unique media types... ", end="")
    dt = t()
    media_set = set(itertools.chain.from_iterable(medias_task.compute()))
    if verbose > 0: print(f"done ({t() - dt}s)")
    media_types_mapping = dict((o, idx) for idx, o in enumerate(media_set))
    del media_set
    gc.collect()

    ddf['hashtags'] = ddf['hashtags'].fillna("")
    ddf['hashtags'] = ddf['hashtags'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)
    hashtags_task = ddf.map_partitions(lambda x: (list(set([a for b in x.hashtags.tolist() for a in b]))), meta=list)
    if verbose > 0: print("Now computing unique hashtag types... ", end="")
    dt = t()
    hashtags_set = set(itertools.chain.from_iterable(hashtags_task.compute()))
    if verbose > 0: print(f"done ({t() - dt}s)")
    hashtags_types_mapping = dict((o, idx) for idx, o in enumerate(hashtags_set))
    del hashtags_set
    gc.collect()

    ddf['links'] = ddf['links'].fillna("")
    ddf['links'] = ddf['links'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)
    links_task = ddf.map_partitions(lambda x: (list(set([a for b in x.links.tolist() for a in b]))), meta=list)
    if verbose > 0: print("Now computing unique link types... ", end="")
    dt = t()
    links_set = set(itertools.chain.from_iterable(links_task.compute()))
    if verbose > 0: print(f"done ({t() - dt}s)")
    links_types_mapping = dict((o, idx) for idx, o in enumerate(links_set))
    del links_set
    gc.collect()

    ddf['domains'] = ddf['domains'].fillna("")
    ddf['domains'] = ddf['domains'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)
    domains_task = ddf.map_partitions(lambda x: (list(set([a for b in x.domains.tolist() for a in b]))), meta=list)
    if verbose > 0: print("Now computing unique domain types... ", end="")
    dt = t()
    domains_set = set(itertools.chain.from_iterable(domains_task.compute()))
    if verbose > 0: print(f"done ({t() - dt}s)")
    domains_types_mapping = dict((o, idx) for idx, o in enumerate(domains_set))
    del domains_set
    gc.collect()


    if convert_tweet_id_to_number:
        ddf["tweet_id"] = ddf[["tweet_id"]].map_partitions(partition_indexing, meta=pd.Series(dtype=np.uint32))
        ddf["tweet_id"] = ddf["tweet_id"].astype(np.uint32)

    if add_bert_length_as_feature:
        ddf["bert_length"] = ddf.map_partitions(_column_length, "bert_base_multilingual_cased_tokens", meta=pd.Series(dtype=np.uint32))

    if seperate_bert or fully_drop_bert:
        if not fully_drop_bert:
            bert_dir = join(output_dir, "bert")
            bert_ddf = ddf[["bert_base_multilingual_cased_tokens", "tweet_id"]]
            bert_ddf.to_parquet(bert_dir)
        ddf = ddf.drop("bert_base_multilingual_cased_tokens", axis="columns")

    if drop_tweet_id:
        ddf = ddf.drop("tweet_id", axis="columns")

    #Remove empties
    ddf['reply']   = ddf['reply'].fillna(0)
    ddf['retweet'] = ddf['retweet'].fillna(0)
    ddf['retweet_comment'] = ddf['retweet_comment'].fillna(0)
    ddf['like']    = ddf['like'].fillna(0)

    #Change dtypes
    ddf["timestamp"] = ddf["timestamp"].astype(np.uint32)
    ddf["a_follower_count"] = ddf["a_follower_count"].astype(np.uint32)
    ddf["a_following_count"] = ddf["a_following_count"].astype(np.uint32)
    ddf["a_account_creation"] = ddf["a_account_creation"].astype(np.uint32)
    ddf["b_follower_count"] = ddf["b_follower_count"].astype(np.uint32)
    ddf["b_following_count"] = ddf["b_following_count"].astype(np.uint32)
    ddf["b_account_creation"] = ddf["b_account_creation"].astype(np.uint32)
    ddf['reply'] = ddf['reply'].astype(np.uint32)
    ddf['retweet'] = ddf['retweet'].astype(np.uint32)
    ddf['retweet_comment'] = ddf['retweet_comment'].astype(np.uint32)
    ddf['like'] = ddf['like'].astype(np.uint32)

    ddf["type"] = ddf["type"].map_partitions(lambda x: x.apply(lambda y: __type_mapping[str(y)]), meta = pd.Series(dtype=np.uint32))

    ddf['links'] = ddf.map_partitions(_map_list_column, links_types_mapping, "links", meta=pd.Series(dtype=object))

    ddf['domains'] = ddf.map_partitions(_map_list_column, domains_types_mapping, "domains", meta=pd.Series(dtype=object))

    ddf['hashtags'] = ddf.map_partitions(_map_list_column, hashtags_types_mapping, "hashtags", meta=pd.Series(dtype=object))

    ddf['medias'] = ddf.map_partitions(_map_list_column, media_types_mapping, "medias", meta=pd.Series(dtype=object))

    if verbose > 0: print("Now executing main task... ", end="")
    dt = t()
    for idx, item in enumerate(ddf.partitions):
        item = item.compute()
        item.to_parquet(join(nobert_dir, f"part-{idx:05}.parquet"))
    if verbose > 0: print(f"done ({t() - dt}s)")
    if verbose > 0: print(f"Finished function ({t() - st}s total)")
