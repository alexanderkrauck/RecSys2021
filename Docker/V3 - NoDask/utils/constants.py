import numpy as np
import os
from collections import Counter
from typing import Tuple, List, Dict
import pandas as pd


COMP_DIR = os.getcwd()
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
FILE_DIR = os.path.realpath(__file__)

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

target_track = set(all_labels)

dtypes_of_features = {
    "bert_base_multilingual_cased_tokens": str,
    "hashtags": str,
    "tweet_id": str,
    "medias": str,
    "links": str,
    "domains": str,
    "type": str,
    "language": str,
    "timestamp": int,
    "a_user_id": str,
    "a_follower_count": np.uint32,
    "a_following_count": np.uint32,
    "a_is_verified": bool,
    "a_account_creation": int,
    "b_user_id": str,
    "b_follower_count": np.uint32,
    "b_following_count": np.uint32,
    "b_is_verified": bool,
    "b_account_creation": int,
    "a_follows_b": bool,
    "reply": np.uint32,
    "retweet": np.uint32,
    "retweet_comment": np.uint32,
    "like": np.uint32
}


def __add_feature_to_label_registry(feature_name: str):
    global target_track
    target_track.add(feature_name)


def any_labels(feature_name: str, cols: Tuple[str, List]) -> bool:
    '''

    Parameters
    ----------
    feature_name
        the name of the new feature
    cols
        the columns it depends on (or one column)
    Returns
    -------
        true if the feature is target derived
    '''
    if type(cols) is str:
        if cols in target_track:
            __add_feature_to_label_registry(feature_name)
            return True
        else:
            return False
    elif type(cols) is list:
        if any([col in target_track for col in cols]):
            __add_feature_to_label_registry(feature_name)
            return True
        else:
            return False
    else:
        raise ValueError("Argument for any_labels should be either string or a list.")


__type_mapping = {"Retweet": 0, "Quote":1, "Reply":2, "TopLevel":3}

__media_type_mapping = {"Photo":0, "Video":1, "GIF":2, "" :4}

__language_mapping = pd.read_csv(os.path.join(ROOT_DIR,"language_mappings.csv"), index_col="language_id")["encode"].to_dict()

all_columns = all_features + all_labels

