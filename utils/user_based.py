import os
import pandas as pd
import numpy as np
import gc

from os.path import join
from pathlib import Path
from datetime import datetime
import time


from .constants import dtypes_of_features, all_columns, all_features, user_centric_cols, __type_mapping

md__ = 2**64

def extract_user_information(
    data_location_folder:str, 
    extraction_folder:str, 
    cutoff_timestamp:int = int(datetime(2021, 2, 19, 0).timestamp())
    ):
    """Extracts user information from the given data
    
    Parameters
    ----------
    data_location_folder: str
        Where the data is stored in csv/tsv format
    extraction_folder: str
        The folder to use for the extracted user data
    cutoff_timestamp: int
        If supplied, only data that has information from before this timestamp is used. This includes all fields (including label fields)
        If None then all data is used. Default is exactly 2 weeks in training data before cutoff.
    """


    Path(extraction_folder).mkdir(exist_ok=True, parents=True)


    files = os.listdir(data_location_folder)
    delta = 0
    last_time = time.time()

    print(f"\nExtracting user information from {len(files)} files \n")
    for i, file in enumerate(files):
        if ".csv" not in file and ".tsv" not in file:
            continue
        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Reading CSV {file}...", end="")
        df = pd.read_csv(join(data_location_folder, file), sep='\x01', header=None, names=all_columns, 
            dtype={k: v for k, v in dtypes_of_features.items() if k in all_features}, usecols=user_centric_cols)


        if cutoff_timestamp is not None:
            print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Applying filters for {file}...", end="")
            df = df[
                    (df["timestamp"] < cutoff_timestamp) & 
                    ((df["reply"] < cutoff_timestamp) | (df["reply"].isnull())) &
                    ((df["retweet"] < cutoff_timestamp) | (df["retweet"].isnull())) &
                    ((df["like"] < cutoff_timestamp) | (df["like"].isnull())) &
                    ((df["retweet_comment"] < cutoff_timestamp) | (df["retweet_comment"].isnull()))
                ]

        
        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Creating User Maps for {file}...", end="")
        df["a_user_id"] = df["a_user_id"].apply(lambda x: int(x, base=16)%md__).astype(np.uint64)
        df["b_user_id"] = df["b_user_id"].apply(lambda x: int(x, base=16)%md__).astype(np.uint64)

        user_dfs = []
        
        cols = ["user_id", "follower_count", "following_count", "verified", "account_creation", "timestamp", "type", "action_type"]


        df_a = df[["a_user_id", "a_follower_count", "a_following_count", "a_is_verified", "a_account_creation", "timestamp", "type"]].copy()
        df_a.loc[:,"action_type"] = "n_present_a_"+df_a["type"]
        df_a.columns = cols
        df_a["day"] = df_a["timestamp"].apply(lambda x: datetime.fromtimestamp(x).timetuple().tm_yday).astype(np.uint16)#day of year
        user_dfs.append(df_a)

        df_b = df[["b_user_id", "b_follower_count", "b_following_count", "b_is_verified", "b_account_creation", "timestamp", "type"]].copy()
        df_b.loc[:,"action_type"] = "n_present_b_"+df_a["type"]
        df_b.columns = cols
        user_dfs.append(df_b)

        for idx, col in enumerate(['reply',"retweet","retweet_comment","like"]):
            #userb_encode
            temp_df = df[["b_user_id", "b_follower_count", "b_following_count", "b_is_verified", "b_account_creation",col, "type"]].copy()
            temp_df = temp_df.dropna(subset=[col])
            temp_df.loc[:,"action_type"] = "n_"+col+"_b_"+temp_df["type"]
            temp_df.columns = cols
            temp_df["day"] = temp_df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).timetuple().tm_yday).astype(np.uint16)#day of year
            user_dfs.append(temp_df)
            #usera_encode
            temp_df = df[["a_user_id", "a_follower_count", "a_following_count", "a_is_verified", "a_account_creation",col, "type"]].copy()
            temp_df = temp_df.dropna(subset=[col])
            temp_df.loc[:,"action_type"] = "n_"+col+"_a_"+temp_df["type"]
            temp_df.columns = cols
            user_dfs.append(temp_df)


        user_df = pd.concat(user_dfs)
        gb = user_df.groupby("user_id")
        gb_cnt = user_df.groupby(["user_id", "action_type"])
        gb_day_cnt = user_df.groupby(["user_id", "day"])

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Extracting Features for {file}...", end="")


        res = gb.agg({
            'follower_count': "first", 
            'following_count':'first', 
            'verified':'first', 
            'account_creation': "first"
            })

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Extracting Counts for {file}...", end="")

        cnt_res = gb_cnt.size().unstack(fill_value=0)
        day_cnt = gb_day_cnt.size().unstack(fill_value=0)
        day_cnt.columns = ["n_day_"+str(int(a)) for a in day_cnt.columns]

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Merging {file}...", end="")

        user_df = pd.merge(res, cnt_res, how='inner', left_index=True, right_index=True)
        user_df = pd.merge(user_df, day_cnt, how="inner", left_index=True, right_index=True)
        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Writing File {file}...", end="")
        user_df.to_parquet(join(extraction_folder, file+".parquet"))
        gc.collect()

        delta = time.time() - last_time
        last_time = time.time()

    print(f"\nSaved all extraction files to {extraction_folder}!")


def create_user_index(
    temp_extraction_folder: str, 
    index_file_name: str
    ):
    """Create an user index from the extracted user data
    
    Parameters
    ----------
    temp_extraction_folder: str
        The folder in which the extracted user data is currently stored
    index_file_name: str
        The file name to use to store the final result
    """

    big_user_df = None
    files = os.listdir(temp_extraction_folder)
    delta = 0
    last_time = time.time()

    print(f"\nCreating index from {len(files)} files \n")
    for i, file in enumerate(files):

        if ".parquet" not in file:
            continue

        df = pd.read_parquet(join(temp_extraction_folder,file))
        for col in df.columns:
            if col.startswith("n_"):
                df[col] = df[col].astype(np.uint16)

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Reading temp file {file}...", end="")
        if big_user_df is None:
            big_user_df = df
            continue

        user_df = df
        user_df.columns = ["next_"+a for a in user_df.columns]

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Collecting overlaps of {file}...", end="")
        from_left = big_user_df.loc[big_user_df.index.difference(user_df.index)]
        from_right = user_df.loc[user_df.index.difference(big_user_df.index)]
        both_extracted = pd.merge(big_user_df, user_df, how="inner", left_index=True, right_index=True)

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Accumulating Features {file}...", end="")
        for col in both_extracted.columns:
            if col.startswith("n_"):
                if str("next_"+col) in both_extracted.columns:#safety
                    both_extracted[col] = both_extracted[col]  +  both_extracted["next_"+col]


        from_right.columns = [a[5:] for a in from_right.columns]
        both_extracted = both_extracted[from_left.columns]

        print(f"\r[{i+1}/{len(files)}][{delta:.2f} s/it]Final Concat with sort {file}...", end="")
        big_user_df = pd.concat([from_left, from_right, both_extracted])
        big_user_df = big_user_df.sort_index()
        gc.collect()

        delta = time.time() - last_time
        last_time = time.time()
    print(f"\nWriting User Index to {index_file_name}!")
    big_user_df.to_parquet(index_file_name)