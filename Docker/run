#!/env/bin/python

location = "test"

feature_columns = ['a_is_verified', 'b_is_verified',
       'a_follows_b', 'bert_token_len',
        'n_photos', 'n_videos', 'n_gifs',
       'type_encoding', 'language_encoding', 'a_followers', 'a_following',
       'b_followers', 'b_following', 'day_of_week', 'hour_of_day',
       'b_creation_delta', 'a_creation_delta', 'TE_language_encoding_has_reply',
       'TE_language_encoding_has_like',
       'TE_language_encoding_has_retweet_comment',
       'TE_language_encoding_has_retweet', 'TE_type_encoding_has_reply',
       'TE_type_encoding_has_like', 'TE_type_encoding_has_retweet_comment',
       'TE_type_encoding_has_retweet']

import utils.compute_and_front as compute
import utils.preprocess as preprocess
import utils.config as config
import utils.model
from utils.dataloader import RecSys2021PandasDataLoader
from dask.distributed import progress, wait
import pandas as pd
import os

if __name__ == '__main__':
        config = config.load_config("config.yaml")


        ddf, delayed_dump, other_delayed = preprocess.preprocess(config)

        with compute.get_dask_compute_environment() as client:
                f = client.persist(delayed_dump)
                progress(f)
                wait(f)

        feature_dir = "preprocessed_validation_features"

        recsysxgb = utils.model.RecSysXGB1("xgb_models_01")

        for part in os.listdir(feature_dir):
                if part != "_common_metadata" and part != "_metadata":
                        data = pd.read_parquet(os.path.join(feature_dir,part))
                        dl = RecSys2021PandasDataLoader(data, feature_columns)
                        recsysxgb.evaluate_test_set(dl, output_file = "results.csv")
        print("done!")
