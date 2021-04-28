import numpy as np
import os
from collections import Counter
from typing import Tuple, List, Dict

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


def is_label(col_name: str) -> bool:
    return col_name in all_labels


__type_mapping = {"Retweet": 0, "Quote":1, "Reply":2, "TopLevel":3}

__media_type_mapping = {"Photo":0, "Video":1, "GIF":2, "" :4}

converters_for_the_original_dataset = {
    #"tweet_id": lambda tid: int(tid[0:8], 16), # take only the last 8 digits, overlap is still pretty unlikely, backwards unhashing is easy
    # TODO: do we really want this, though ?
}

all_columns = all_features + all_labels

COMP_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
FILE_DIR = os.path.realpath(__file__)