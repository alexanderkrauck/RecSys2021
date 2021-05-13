import numpy as np
from collections import Counter
from typing import Tuple
from .constants import __media_type_mapping, __type_mapping, __language_mapping

import dask.dataframe as dd
from typing import List

#from transformers import BertTokenizer

from datetime import datetime

#tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def save_log(item):
    return np.log(abs(item) + 1)

def quantile_mapping(x):
    if x < 421:
        return 1
    if x < 1353:
        return 2
    if x < 4945:
        return 3
    if x < 30774:
        return 4
    return 5

div = 2**32
single_column_features = {
    # TODO: apply log to numerical values (or not?)
    # name of the feature : ([required columns], function, output type,
    #   apply level (1: series lambda/apply, 2: df row lambda, 3: df map_partitions))
    # also highly recommended to specify at least one of the features from all_labels in the requirements even if not
    #   needed for computation for all target derived features.
    "bert_token_len": ('bert_base_multilingual_cased_tokens', lambda bertenc: save_log(len(bertenc.split('\t'))), np.uint32, 1),
    "has_reply": ('reply', lambda v: v > 0., bool, 3),
    "has_retweet": ('retweet', lambda v: v > 0., bool, 3),
    "has_retweet_comment": ('retweet_comment', lambda v: v > 0., bool, 3),
    "has_like": ('like', lambda v: v > 0., bool, 3),
    "n_photos": ('medias', lambda v: save_log(Counter(v.split('\t'))['Photo']) if v else 0, np.float32, 1),
    "n_videos": ('medias', lambda v: save_log(Counter(v.split('\t'))['Video']) if v else 0, np.float32, 1),
    "n_gifs": ('medias', lambda v: save_log(Counter(v.split('\t'))['GIF']) if v else 0, np.float32, 1),
    "reply_age": (['reply', 'timestamp', 'has_reply'], lambda df: (df['reply']-df['timestamp'])*df['has_reply'], np.uint32, 3),
    "like_age": (['like', 'timestamp', 'has_like'], lambda df: (df['like']-df['timestamp'])*df['has_like'], np.uint32, 3),
    "retweet_age": (['retweet', 'timestamp', 'has_retweet'], lambda df: (df['retweet']-df['timestamp'])*df['has_retweet'], np.uint32, 3),
    "retweet_comment_age": (['retweet_comment', 'timestamp', 'has_retweet_comment'], lambda df: (df['retweet_comment']-df['timestamp'])*df['has_retweet_comment'], np.uint32, 3),
    #"bert_char_len": ('bert_base_multilingual_cased_tokens', lambda bertenc: len(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(bertenc.split("\t")))), np.uint16, 1), #SLOW
    "type_encoding": ('type', lambda x: __type_mapping[x], np.uint8, 1),
    "language_encoding": ('language', lambda x: __language_mapping[x], np.uint16, 1),
    "a_followers" :('a_follower_count', save_log, np.float32, 1),
    "a_following" :('a_following_count', save_log, np.float32, 1),
    "b_followers" :('b_follower_count', save_log, np.float32, 1),
    "b_following" :('b_following_count', save_log, np.float32, 1),
    "day_of_week" :('timestamp', lambda x: datetime.fromtimestamp(x).weekday(), np.uint8, 1),
    "hour_of_day":('timestamp', lambda x: datetime.fromtimestamp(x).hour, np.uint8, 1),
    "b_creation_delta": (['timestamp', 'b_account_creation'], lambda x: save_log(x["timestamp"] - x["b_account_creation"]), np.float32, 3),
    "a_creation_delta": (['timestamp', 'a_account_creation'], lambda x: save_log(x["timestamp"] - x["a_account_creation"]), np.float32, 3),
    "tweet_hash": ("tweet_id", lambda x: int(x, 16)%div , np.uint32, 1),#this introduces a slight error but should be fine with the big picture...
    "b_hash": ("b_user_id", lambda x: int(x, 16)%div , np.uint32, 1),
    "a_hash": ("a_user_id", lambda x: int(x, 16)%div , np.uint32, 1),
    "quantile": ("a_follower_count", lambda x: quantile_mapping(x), np.uint8, 1)
}


def TE_get_name(feature_name: str, target_name: str) -> str:
    return 'TE_'+feature_name+'_'+target_name


def TE_dataframe_dask(df: dd.DataFrame,
                      feature_name: str,
                      target_name: str,
                      w_smoothing: int = 20,
                      dt=np.float64,
                      counts_and_means: dd.DataFrame = None) -> Tuple[dd.DataFrame]:
    if counts_and_means is None:
        counts_and_means = df[[target_name, feature_name]].groupby(feature_name)[target_name].agg(['count', 'mean'])
        counts_and_means["total_mean"] = df[target_name].mean() # redundant but necessary
    TE_map = counts_and_means.apply(lambda cm: (cm["count"]*cm["mean"]+w_smoothing*cm["total_mean"])/(cm["count"]+w_smoothing),
                                    axis=1,
                                    meta=(TE_get_name(feature_name, target_name), dt)
                                    )
    # the only way to vectorize this, joining on non-index - must be somewhat costly
    df = df.join(TE_map, on=feature_name, how='left')
    df = df.fillna(-1)#for test set...

    return df, counts_and_means    # all lazy all dask, should be fine, evaluated in the end when all the things are merged


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
