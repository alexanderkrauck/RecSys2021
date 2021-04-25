from utils.download import decompress_lzo_file

from dask import dataframe as dd
from dask.distributed import Client, progress
from dask import delayed

import numpy as np
import pandas as pd

from collections import Counter
from typing import Union, List, Tuple, Callable, Dict
import os
from os.path import join
import yaml

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
# TODO: make the following 4 configurable and change the one above to getcwd() (might come quite handy on the server)
REL_COMP_DIR = "compressed_data"
REL_UNCOMP_DIR = "uncompressed_data"
REL_PREPRO_DIR = "preprocessed"
# split them from the original features so that we dont have to rewrite the whole dataset to the disk every time
REL_NEW_FEATURES_DIR = "preprocessed_features"

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
                "a_follows_b"]

all_labels = ["reply",
              "retweet",
              "retweet_comment",
              "like"]

dtypes_of_features = {
    "bert_base_multilingual_cased_tokens": str,
    "hashtags": str,
    "tweet_id": str,
    "medias": str,
    "links": str,
    "domains": str,
    "type": str,
    "language": str,
    "timestamp": np.uint32,
    "a_user_id": str,
    "a_follower_count": np.uint32,
    "a_following_count": np.uint32,
    "a_is_verified": bool,
    "a_account_creation": np.uint32,
    "b_user_id": str,
    "b_follower_count": np.uint32,
    "b_following_count": np.uint32,
    "b_is_verified": bool,
    "b_account_creation": np.uint32,
    "a_follows_b": bool,
    "reply": np.uint32,
    "retweet": np.uint32,
    "retweet_comment": np.uint32,
    "like": np.uint32
}

converters_for_the_original_dataset = {
    "tweet_id": lambda tid: int(tid[0:16], 16)
    # TODO: convert other fields with hashes in them into integer types
}

all_columns = all_features + all_labels

single_column_features = {
    # TODO: would be nice to rehash userid to uint64 and other identifiers (especially 'language')
    # TODO: apply log to numerical values (or not?)
    # name of the feature : ([required columns], function, output type,
    #   apply level (1: series lambda/apply, 2: df row lambda, 3: df map_partitions))
    "bert_token_len": ('bert_base_multilingual_cased_tokens', lambda bertenc: len(bertenc.split('\t')), np.uint32, 1),
    "has_reply": ('reply', lambda v: v > 0., bool, 3),
    "has_retweet": ('retweet', lambda v: v > 0., bool, 3),
    "has_retweet_comment": ('retweet_comment', lambda v: v > 0., bool, 3),
    "has_like": ('like', lambda v: v > 0., bool, 3),
    "n_photos": ('medias', lambda v: Counter(v.split('\t'))['Photo'] if v else 0, np.uint8, 1),  # FIXME: filter the stuff more in the first stage
    "n_videos": ('medias', lambda v: Counter(v.split('\t'))['Video'] if v else 0, np.uint8, 1),
    "n_gifs": ('medias', lambda v: Counter(v.split('\t'))['GIF'] if v else 0, np.uint8, 1),
    "reply_age": (['reply', 'timestamp', 'has_reply'], lambda df: (df['reply']-df['timestamp'])*df['has_reply'], np.uint32, 3),
    "like_age": (['like', 'timestamp', 'has_like'], lambda df: (df['like']-df['timestamp'])*df['has_like'], np.uint32, 3),
    "retweet_age": (['retweet', 'timestamp', 'has_retweet'], lambda df: (df['retweet']-df['timestamp'])*df['has_retweet'], np.uint32, 3),
    "retweet_comment_age": (['retweet_comment', 'timestamp', 'has_retweet_comment'], lambda df: (df['retweet_comment']-df['timestamp'])*df['has_retweet_comment'], np.uint32, 3),
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
    # the only way to vectorize this, joining on non-index - must be somewhat costly
    df = df.join(TE_map, on=feature_name, how='left')

    return df    # all lazy all dask, should be fine, evaluated in the end when all the things are merged


def conditional_probabilities_as_per_config(df: dd.DataFrame,
                                            config: dict = None):
    if not config:
        config = load_default_config()

    verbosity = config['verbose']
    features = config['marginal_prob_columns']['features']
    labels = config['marginal_prob_columns']['per_labels']

    with get_dask_compute_environment(config) as client:
        delayed_series = conditional_probabilities(df, features, labels)
        delayed_op = dd.to_csv(delayed_series, compute=False)
        f = client.persist(delayed_op)
        if verbosity > 0:
            progress(f)
        f.compute()

    return


def conditional_probabilities(df: dd.DataFrame,
                              features: List[str],
                              labels: List[str]) -> dd.Series:
    '''

    Parameters
    ----------
    df
        Dask Dataframe with the data
    features
        the categorical feature labels that are to be included in the computation
    labels
        the categorical labels that are to be included in the computation

    Returns
    -------
        A lazy series, indexed on values of features and labels and contains the conditional probabilities of
            the features.
    '''
    # FIXME: does not support supplying new features only due the usage of tweet_id, find or make another column that
    #   works well as an omnipresent pivot
    marg = df[labels+["tweet_id"]].groupby(labels)["tweet_id"].count().rename('marginal').to_frame() # because tweet_id is always there
    joint = df[labels+features+['tweet_id']].groupby(labels+features)['tweet_id'].count().rename('joint').to_frame()
    lp = joint.join(marg, on=labels, how='left')
    pp = lp.joint / lp.marginal
    return pp


def load_all_preprocessed_data(
        only_new_features=False,
        prepro_dir: Union(os.PathLike, str) = join(ROOT_DIR, REL_PREPRO_DIR),
        new_features_dir: Union(os.PathLike, str) = join(ROOT_DIR, REL_NEW_FEATURES_DIR)
    ) -> dd.DataFrame:
    '''

    Parameters
    ----------
    only_new_features
        loading the whole thing might prove pretty heavy, sometimes loading just the preprocessed features is just
            as fine
    Returns
    -------
        A lazy dataframe that has the original preprocessed features and new computed features loaded
    '''
    if not only_new_features:
        original_df = dd.read_parquet(prepro_dir)
        extra_features_df = dd.read_parquet(new_features_dir)
        return dd.merge(original_df, extra_features_df, how='inner', left_index=True, right_index=True)  # on index, so fast
    else:
        return dd.read_parquet(new_features_dir)


def load_default_config(root_dir: Union(os.PathLike, str) = ROOT_DIR) -> dict:
    with open(os.path.join(root_dir, "config.yaml")) as f:
        return yaml.load(f, Loader=yaml.CLoader)


def get_dask_compute_environment(config: dict = None) -> Client:
    '''

    Parameters
    ----------
    config
        config.yaml
    Returns
    -------
        a client object that is already set as default in the current context
    '''
    if not config:
        config = load_default_config()

    verbosity = config['verbose']
    client_n_workers = config['n_workers']
    client_n_threads = config['n_threads_per_worker']
    client_memlim = config['mem_lim']

    # initialize a client according to the configuration and make it local
    client = Client(memory_limit=client_memlim,
                    n_workers=client_n_workers,
                    threads_per_worker=client_n_threads,
                    processes=True)
    client.as_current()     # this assigns all the operations implicitly to this client, I believe.

    if verbosity >= 1:
        print("Compute environment established: {}".format(client))

    return client


def preprocess(
        config: dict = None,
        root_dir: Union(os.PathLike, str) = ROOT_DIR,
        comp_dir: Union(os.PathLike, str) = None,
        uncomp_dir: Union(os.PathLike, str) = None,
        new_features_dir: Union(os.PathLike, str) = None,
        prepro_dir: Union(os.PathLike, str) = None 
    ) -> Tuple[dd.DataFrame, delayed]:
    '''

    Parameters
    ----------
    config: dict
        configuration dictionary from the yaml file
    comp_dir: Union(os.PathLike, str)
        The location of the compressed data.

    
    Returns
    -------
        lazy dataframe that contains all of the features, can be used for more computations if desired
        delayed object that needs to be computed in an environment of choice in order for the files to be produced
        e.g.
        f = client.persist(futures_dump)
        progress(f)
        f.compute()     # gathers the result
    '''

    
    #Set default directories if not specified (based on root dir)
    if not comp_dir: comp_dir = join(ROOT_DIR, REL_COMP_DIR)
    if not uncomp_dir: uncomp_dir = join(ROOT_DIR, REL_UNCOMP_DIR)
    if not new_features_dir: new_features_dir = join(ROOT_DIR, REL_NEW_FEATURES_DIR)
    if not prepro_dir: prepro_dir = join(ROOT_DIR, REL_PREPRO_DIR)
    
    #Load default config if not specified and extract required parameters
    if not config: config = load_default_config(root_dir)
    verbosity = config['verbose']
    data_source = config['load_from']

    ddf = None
    if data_source == 'comp' or data_source == "uncomp":
        if datasource == 'comp':
            # decompress lzo files to uncomp directory
            decompress_lzo_file(comp_dir, uncomp_dir, delete_compressed=False, overwrite=False, verbose=verbosity)

        # start with reading the files lazily and locally, assume they are uncompressed
        unpacked_files = [os.path.join(uncomp_dir, f)
                          for f in os.listdir(uncomp_dir) if os.path.isfile(os.path.join(uncomp_dir, f))]
        if verbosity >= 2:
            print(unpacked_files)

        ddf = dd.read_csv(unpacked_files, sep='\x01', header=None, names=all_columns, blocksize="128MB",
                          dtype={},
                          converters={}
                          )  # TODO: add dtypes from above and converters

        ddf["idx"] = 1
        ddf["idx"] = ddf["idx"].cumsum()
        ddf = ddf.set_index("idx")

        # do some basic maintenance of the dataset
        # TODO: convert bert encoding field etc.
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

        with get_dask_compute_environment(config) as client:
            # now drop the resulting dataframe to the location where we can find it again
            futures_dump = ddf.to_parquet(PREPRO_DIR, compute=False)
            # works under assumption that the local machine shares the file system with the dask client
            f = client.persist(futures_dump)
            if verbosity > 0:
                progress(f)
            f.compute()

        if verbosity >= 1:
            print("The uncompressed dataset is saved to the disk.")

        # free up the memory
        del ddf, f

    # default parameters work just fine
    ddf = dd.read_parquet(prepro_dir)
    original_cols = [col for col in ddf.columns]

    # first add the features as per configuration file
    sc_features_series = []
    for feature in config['basic_features']:
        cols, fun, dt, iterlevel = single_column_features[feature]
        if iterlevel == 1:
            f_series = ddf[cols].apply(fun, meta=(feature, dt))
        elif iterlevel == 2:
            f_series = ddf[cols].to_frame().apply(fun, axis=1, meta=(feature, dt))
        elif iterlevel == 3:
            f_series = ddf[cols].map_partitions(fun, meta=(feature, dt))
        else:
            raise ValueError("Wrong iterlevel.")
        ddf = dd.merge(ddf,
                       f_series,
                       how='inner',
                       left_index=True,
                       right_index=True)

    # then add the TEs as per configuration
    for te_feature, te_target in config['TE_features'].items():
        ddf = TE_dataframe_dask(ddf, te_feature, te_target)       # already joins in the function

    new_feature_columns = [col for col in ddf.columns if col not in original_cols]
    if verbosity >= 1:
        print("The following preprocessed columns are dumped: ", new_feature_columns)
    delayed_dump = ddf[new_feature_columns].to_parquet(new_features_dir, compute=False)

    if verbosity >= 1:
        print("Feature generation ready, associated delayed objects: {}\n\n{}.".format(ddf, delayed_dump))

    return ddf, delayed_dump

