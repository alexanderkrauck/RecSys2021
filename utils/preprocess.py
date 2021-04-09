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

__media_type_mapping = {"Photo":0, "Video":1, "GIF":2, "" :4}

#Helper Functions
def _partition_indexing(df, partition_info=None):
    df["tweet_id"] = 1
    df["tweet_id"] = df["tweet_id"].cumsum()
    #This is not a stable choice but will work well with 64MB data splits
    df["tweet_id"] = df["tweet_id"] + partition_info["number"] * 200000 
    return df["tweet_id"]

def _collect_unique_items(df):
    hashtags = list(set([a for b in df["hashtags"].tolist() for a in b]))
    links = list(set([a for b in df["links"].tolist() for a in b]))
    domains = list(set([a for b in df["domains"].tolist() for a in b]))
    languages = list(set(df["language"].tolist()))

    return (hashtags, links, domains, languages)

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


    ddf['hashtags'] = ddf['hashtags'].fillna("")
    ddf['hashtags'] = ddf['hashtags'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)

    ddf['links'] = ddf['links'].fillna("")
    ddf['links'] = ddf['links'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)

    ddf['domains'] = ddf['domains'].fillna("")
    ddf['domains'] = ddf['domains'].map_partitions(lambda x: list(x.str.split("\t")), meta=list)


    if verbose > 0: print("Now computing unique column elements... ", end="")
    dt = t()
    result_tuples = ddf.map_partitions(_collect_unique_items, meta=tuple).compute()

    unique_hashtags = set(itertools.chain.from_iterable([result_tuple[0] for result_tuple in result_tuples]))
    hashtags_types_mapping =  dict((o, idx) for idx, o in enumerate(unique_hashtags))
    unique_links = set(itertools.chain.from_iterable([result_tuple[1] for result_tuple in result_tuples]))
    links_types_mapping =  dict((o, idx) for idx, o in enumerate(unique_links))
    unique_domains = set(itertools.chain.from_iterable([result_tuple[2] for result_tuple in result_tuples])) 
    domains_types_mapping =  dict((o, idx) for idx, o in enumerate(unique_domains))
    unique_languages = set(itertools.chain.from_iterable([result_tuple[3] for result_tuple in result_tuples])) 
    language_types_mapping =  dict((o, idx) for idx, o in enumerate(unique_languages))
    print("done")

    del result_tuples, unique_hashtags, unique_links, unique_domains, unique_languages
    gc.collect()


    if convert_tweet_id_to_number:
        ddf["tweet_id"] = ddf[["tweet_id"]].map_partitions(partition_indexing, meta=pd.Series(dtype=np.uint32))
        ddf["tweet_id"] = ddf["tweet_id"].astype(np.uint32)


    #From here we act on the individual partitions
    if verbose > 0: print(f"Now outputting preprocessed tsv files to {nobert_dir}")
    dt=t()
    for idx, df in enumerate(ddf.partitions):#TODO: Here it would be possible to add multiprocessing
        df = df.compute()
        if add_bert_length_as_feature:
            df["bert_length"] = df["bert_base_multilingual_cased_tokens"].apply(lambda x: len(str(x).split("\t")))
        if seperate_bert or fully_drop_bert:
            if not fully_drop_bert:
                bert_dir = join(output_dir, "bert")
                Path(bert_dir).mkdir(parents=True, exist_ok=True)
                bert_df = df[["bert_base_multilingual_cased_tokens", "tweet_id"]]
                bert_df.to_parquet(join(bert_dir), f"part-{idx:05}.parquet")
            df = df.drop("bert_base_multilingual_cased_tokens", axis="columns")
        if drop_tweet_id:
            df = df.drop("tweet_id", axis="columns")
        #Remove empties
        df['reply']   = df['reply'].fillna(0)
        df['retweet'] = df['retweet'].fillna(0)
        df['retweet_comment'] = df['retweet_comment'].fillna(0)
        df['like']    = df['like'].fillna(0)

        #Change dtypes
        df["timestamp"] = df["timestamp"].astype(np.uint32)
        df["a_follower_count"] = df["a_follower_count"].astype(np.uint32)
        df["a_following_count"] = df["a_following_count"].astype(np.uint32)
        df["a_account_creation"] = df["a_account_creation"].astype(np.uint32)
        df["b_follower_count"] = df["b_follower_count"].astype(np.uint32)
        df["b_following_count"] = df["b_following_count"].astype(np.uint32)
        df["b_account_creation"] = df["b_account_creation"].astype(np.uint32)
        df['reply'] = df['reply'].astype(np.uint32)
        df['retweet'] = df['retweet'].astype(np.uint32)
        df['retweet_comment'] = df['retweet_comment'].astype(np.uint32)
        df['like'] = df['like'].astype(np.uint32)

        
        df["type"] = df["type"].map(__type_mapping)

        df["language"] = df["language"].map(language_types_mapping)

        df['links'] = df["links"].apply(lambda x: [links_types_mapping[str(item)] for item in x])

        df['domains'] = df["domains"].apply(lambda x: [domains_types_mapping[str(item)] for item in x])

        df['hashtags'] = df["hashtags"].apply(lambda x: [hashtags_types_mapping[str(item)] for item in x])

        df['medias'] = df["medias"].apply(lambda x: [__media_type_mapping[str(item)] for item in x])

        df.to_parquet(join(nobert_dir, f"part-{idx:05}.parquet"))
        
        if verbose > 0: print(f"\rFinished {idx+1}/{ddf.npartitions}. ({t() - dt} s/it)", end="")
        dt = t()
    if verbose > 0: print()
