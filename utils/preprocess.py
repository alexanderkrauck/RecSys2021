"""
Utility functions for preprocessing files from the RecSys2020 Servers

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "08-03-2021"


import os
from os.path import join
from pathlib import Path

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


#Helper Functions
def _partition_indexing(df, partition_info=None):
    df["tweet_id"] = 1
    df["tweet_id"] = df["tweet_id"].cumsum()
    #This is not a stable choice but will work well with 64MB data splits
    df["tweet_id"] = df["tweet_id"] + partition_info["number"] * 200000 
    return df["tweet_id"]

def initial_preprocess_data(
        csv_path: str,
        output_dir: str,
        convert_tweet_id_to_number: bool,
        seperate_bert: bool,
        fully_drop_bert: bool,
        drop_tweet_id: bool,
        add_bert_length_as_feature: bool = True,
        blocksize: str = "64MB"
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
        TODO: needs to be implemented in a manner that works without of exploding the RAM
    blocksize : str
        The blocksize in which the data of in the path will be split. If this is chosen to high, the memory will become full.
    """

    nobert_dir = join(output_dir,"nobert")
    Path(nobert_dir).mkdir(parents=True, exist_ok=True)


    ddf = dd.read_csv(csv_path, sep='\x01', header=None, names=__all_columns, blocksize="64MB")

    if convert_tweet_id_to_number:
        ddf["tweet_id"] = ddf[["tweet_id"]].map_partitions(partition_indexing, meta=pd.Series(dtype=np.uint32))
        ddf["tweet_id"] = ddf["tweet_id"].astype(np.uint32)

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

    ddf['medias'] = ddf['medias'].fillna("")
    ddf['medias'] = ddf['medias'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)

    ddf['hashtags'] = ddf['hashtags'].fillna("")
    ddf['hashtags'] = ddf['hashtags'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)

    ddf['links'] = ddf['links'].fillna("")
    ddf['links'] = ddf['links'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)

    ddf['domains'] = ddf['domains'].fillna("")
    ddf['domains'] = ddf['domains'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)

    lists_task = ddf.map_partitions(lambda x: (list(set([a for b in x.medias.tolist() for a in b]))), meta=list)
    media_set = set(itertools.chain.from_iterable(lists_task.compute()))
    media_types_mapping = dict((o, idx) for idx, o in enumerate(media_set))
    del media_set
    gc.collect()
    ddf['medias'] = ddf["medias"].map_partitions(lambda x: [media_types_mapping[item] for item in x], meta=list)


    ddf.to_parquet(nobert_dir)