from utils.download import decompress_lzo_file

from dask import dataframe as dd
from dask.distributed import Client, progress

import numpy as np
import pandas as pd

from typing import Union, List, Tuple, Callable, Dict
import os
import yaml

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
# TODO: make the following 4 configurable
COMP_DIR = os.path.join(ROOT_DIR, "compressed_data")
UNCOMP_DIR = os.path.join(ROOT_DIR, "uncompressed_data")
PREPRO_DIR = os.path.join(ROOT_DIR, "preprocessed")
# split them from the original features so that we dont have to rewrite the whole dataset to the disk every time
NEW_FEATURES_DIR = os.path.join(ROOT_DIR, "preprocessed_features")

all_features = ["bert_base_multilingual_cased_tokens",
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
                "a_follows_b"]  # as far as I know from the forum (b always follows a in this dataset according to the forum)

all_labels = ["reply",
              "retweet",
              "retweet_comment",
              "like"]

all_columns = all_features + all_labels

single_column_features = {
    # name of the feature : ([required columns], function, output type)
    "bert_text_len": ('bert_base_multilingual_cased_tokens', lambda bertenc: len(bertenc.split('\t')), np.uint32),
    "has_reply": ('reply', lambda v: v > 0., bool),
    "has_retweet": ('retweet', lambda v: v > 0., bool),
    "has_retweet_comment": ('retweet_comment', lambda v: v > 0., bool),
    "has_like": ('like', lambda v: v > 0., bool),
#TODO:    "post_time_diff": ([''])
}


def TE_dataframe_dask(df: dd.DataFrame,
                      feature_name: str,
                      target_name: str,
                      w_smoothing: int = 20,
                      dt=np.float64) -> dd.Series:

    counts_and_means = df[[target_name, feature_name]].groupby(feature_name)[target_name].agg(['count', 'mean'])
    counts_and_means["total_mean"] = df[target_name].mean()  # redundant but necessary
    TE_map = counts_and_means.apply(lambda cm: (cm["count"]*cm["mean"]+w_smoothing*cm["total_mean"])/(cm["count"]+w_smoothing),
                                    axis=1,
                                    meta=('TE_'+feature_name+'_'+target_name, dt)
                                    )
    # the only way to vectorize this, joining on non-index - must be very costly
    df = df.join(TE_map, on=feature_name, how='left')

    return df    # all lazy all dask, should be fine, evaluated in the end when all the things are merged


def load_default_config() -> dict:
    with open(os.path.join(ROOT_DIR, "config.yaml")) as f:
        return yaml.load(f, Loader=yaml.CLoader)


def preprocess(config: dict = None) -> None:
    if not config:
        config = load_default_config()

    verbosity = config['verbose']
    client_n_workers = config['n_workers']
    client_n_threads = config['n_threads_per_worker']
    client_memlim = config['mem_lim']
    data_source = config['load_from']

    # initialize a client according to the configuration and make it local
    client = Client(memory_limit=client_memlim,
                    n_workers=client_n_workers,
                    threads_per_worker=client_n_threads,
                    processes=True)
    client.as_current()     # this assigns all the operations implicitly to this client, I believe.
    if verbosity >= 0:
        print(client)

    ddf = None
    if data_source == 'comp':
        # decompress lzo files
        decompress_lzo_file(COMP_DIR, UNCOMP_DIR, delete_compressed=False, overwrite=False, verbose=verbosity)

        # start with reading the files lazily and locally, assume they are uncompressed
        unpacked_files = [os.path.join(UNCOMP_DIR, f)
                          for f in os.listdir(UNCOMP_DIR) if os.path.isfile(os.path.join(UNCOMP_DIR, f))]
        if verbosity >= 2:
            print(unpacked_files)

        ddf = dd.read_csv(unpacked_files, sep='\x01', header=None, names=all_columns, blocksize="128MB")

        ddf["idx"] = 1
        ddf["idx"] = ddf["idx"].cumsum()
        ddf = ddf.set_index("idx")

        # do some basic maintenance of the dataset
        ddf["timestamp"] = ddf["timestamp"].astype(np.uint32)
        ddf["a_follower_count"] = ddf["a_follower_count"].astype(np.uint32)
        ddf["a_following_count"] = ddf["a_following_count"].astype(np.uint32)
        ddf["a_account_creation"] = ddf["a_account_creation"].astype(np.uint32)
        ddf["b_follower_count"] = ddf["b_follower_count"].astype(np.uint32)
        ddf["b_following_count"] = ddf["b_following_count"].astype(np.uint32)
        ddf["b_account_creation"] = ddf["b_account_creation"].astype(np.uint32)

        ddf['reply'] = ddf['reply'].fillna(0)
        ddf['retweet'] = ddf['retweet'].fillna(0)
        ddf['retweet_comment'] = ddf['retweet_comment'].fillna(0)
        ddf['like'] = ddf['like'].fillna(0)

        ddf['reply'] = ddf['reply'].astype(np.uint32)
        ddf['retweet'] = ddf['retweet'].astype(np.uint32)
        ddf['retweet_comment'] = ddf['retweet_comment'].astype(np.uint32)
        ddf['like'] = ddf['like'].astype(np.uint32)

        # TODO: would be nice to rehash userid to uint64 and other identifiers (especially 'language')

        # TODO: apply log to numerical values (or not?)


        # now drop the resulting dataframe to the location where we can find it again
        futures_dump = ddf.to_parquet(PREPRO_DIR, compute=False)
        # works under assumption that the local machine shares the file system with the dask client
        f = client.persist(futures_dump)
        progress(f)
        f.compute()

        if verbosity >= 1:
            print("The uncompressed dataset is dumped to the disk.")

        # free up the memory
        del ddf, f

    # default parameters work just fine
    ddf = dd.read_parquet(PREPRO_DIR)
    original_cols = [col for col in ddf.columns]

    # first add the features as per configuration file
    sc_features_series = []
    for feature in config['basic_features']:
        cols, fun, dt = single_column_features[feature]
        f_series = ddf[cols].apply(fun, meta=(feature, dt))#.to_frame()
        ddf = dd.merge(ddf,
                       f_series,
                       how='inner',
                       left_index=True,
                       right_index=True)

    # then add the TEs as per configuration
    for te_feature, te_target in config['TE_features'].items():
        ddf = TE_dataframe_dask(ddf, te_feature, te_target)       # already joins in the function

    # TODO: collect marginal distributions on selected categorical columns

    # TODO: collect mean/std on numerical columns

    new_feature_columns = [col for col in ddf.columns if col not in original_cols]
    if verbosity >= 1:
        print("The following preprocessed columns are dumped: ", new_feature_columns)
    ddf = ddf[new_feature_columns]
    futures_dump = ddf.to_parquet(NEW_FEATURES_DIR, compute=False)
    f = client.persist(futures_dump)
    progress(f)

    if verbosity >= 1:
        print("Dumped to files and done.")

    return
