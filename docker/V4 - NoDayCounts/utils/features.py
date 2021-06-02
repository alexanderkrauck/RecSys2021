import numpy as np
from collections import Counter
from typing import Tuple
from .constants import __media_type_mapping, __type_mapping, __language_mapping

from typing import List

#from transformers import BertTokenizer

from datetime import datetime

#tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

def save_log(item):
    return np.log(abs(item) + 1)

def symlog(item):

    
    sign = np.sign(item)
    item *= sign #ensure positive
    if item <= 1:
        return 0

    return np.log(item) * sign


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
    "n_photos": ('medias', lambda v: save_log(Counter(v.split('\t'))['Photo']) if v else -1, np.float32, 1),
    "n_videos": ('medias', lambda v: save_log(Counter(v.split('\t'))['Video']) if v else -1, np.float32, 1),
    "n_gifs": ('medias', lambda v: save_log(Counter(v.split('\t'))['GIF']) if v else -1, np.float32, 1),
    #"reply_age": (['reply', 'timestamp', 'has_reply'], lambda df: (df['reply']-df['timestamp'])*df['has_reply'], np.uint32, 3),
    #"like_age": (['like', 'timestamp', 'has_like'], lambda df: (df['like']-df['timestamp'])*df['has_like'], np.uint32, 3),
    #"retweet_age": (['retweet', 'timestamp', 'has_retweet'], lambda df: (df['retweet']-df['timestamp'])*df['has_retweet'], np.uint32, 3),
    #"retweet_comment_age": (['retweet_comment', 'timestamp', 'has_retweet_comment'], lambda df: (df['retweet_comment']-df['timestamp'])*df['has_retweet_comment'], np.uint32, 3),
    #"bert_char_len": ('bert_base_multilingual_cased_tokens', lambda bertenc: len(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(bertenc.split("\t")))), np.uint16, 1), #SLOW
    "type_encoding": ('type', lambda x: __type_mapping[x], np.uint8, 1),
    "language_encoding": ('language', lambda x: __language_mapping[x], np.uint16, 1),
    "a_followers" :('a_follower_count', save_log, np.float32, 1),
    "a_following" :('a_following_count', save_log, np.float32, 1),
    "b_followers" :('b_follower_count', save_log, np.float32, 1),
    "b_following" :('b_following_count', save_log, np.float32, 1),
    "day_of_week" :('timestamp', lambda x: datetime.fromtimestamp(x).weekday(), np.uint8, 1),
    "hour_of_day":('timestamp', lambda x: datetime.fromtimestamp(x).hour, np.uint8, 1),
    #"b_creation_delta": (['timestamp', 'b_account_creation'], lambda x: symlog(x["timestamp"] - x["b_account_creation"]), np.float32, 3),
    #"a_creation_delta": (['timestamp', 'a_account_creation'], lambda x: symlog(x["timestamp"] - x["a_account_creation"]), np.float32, 3),
    #"a_b_creation_delta": (['a_account_creation', 'b_account_creation'], lambda x: symlog(x["a_account_creation"] - x["b_account_creation"]), np.float32, 3),
    #"tweet_hash": ("tweet_id", lambda x: int(x, 16)%div , np.uint32, 1),#this introduces a slight error but should be fine with the big picture...
    #"b_hash": ("b_user_id", lambda x: int(x, 16)%div , np.uint32, 1),
    #"a_hash": ("a_user_id", lambda x: int(x, 16)%div , np.uint32, 1),
    "quantile": ("a_follower_count", lambda x: quantile_mapping(x), np.uint8, 1)
}

single_column_targets = {
    "has_reply": ('reply', lambda v: v > 0., bool, 1),
    "has_retweet": ('retweet', lambda v: v > 0., bool, 1),
    "has_retweet_comment": ('retweet_comment', lambda v: v > 0., bool, 1),
    "has_like": ('like', lambda v: v > 0., bool, 1)
}