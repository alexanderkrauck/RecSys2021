from utils.download import decompress_lzo_file

from dask import dataframe as dd
from dask import delayed

import numpy as np
import pandas as pd

from collections import Counter
from typing import Union, List, Tuple, Callable, Dict
import os
from os.path import join
from pathlib import Path

from constants import COMP_DIR
from config import load_feature_config, load_mop_config, load_compute_config, load_manifest, dump_manifest
from compute_and_front import uncompress_and_parquetize
from features import single_column_features, conditional_probabilities, TE_dataframe_dask


def preprocess(
        config: dict = None,
        comp_dir: Union[os.PathLike, str] = COMP_DIR
    ) -> Tuple[dd.DataFrame, dd.DataFrame, List[delayed]]:
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

    # split the config file
    mop_config = load_mop_config(config)  # mode of preprocessing part
    compute_config = load_compute_config(config)   # all to do with computation
    feature_config = load_feature_config(config)     # all to do with specifications of the features that we need

    #Load default config if not specified and extract required parameters
    verbosity = mop_config['verbose']

    # unpack some of the config variables
    train_set_mode = mop_config['train_set']
    data_source = mop_config['load_from']

    #Set default directories if not specified (based on root dir)
    comp_dir = join(comp_dir, mop_config['compressed_dir'])
    uncomp_dir = join(comp_dir, mop_config['uncompressed_dir'])
    new_features_dir = join(comp_dir, mop_config['feature_dir'])
    prepro_dir = join(comp_dir, mop_config['preprocessed_dir'])
    stat_dir = join(comp_dir, mop_config['prepro_statistics_dir'])

    if data_source == 'comp' or data_source == "uncomp":
        uncompress_and_parquetize(config)

    # default parameters work just fine
    ddf = dd.read_parquet(prepro_dir, index="idx")
    other_delayed = []
    manifest = {'available_features': [],
                'TE_stats': {}} if train_set_mode else load_manifest(os.path.join(comp_dir, mop_config['manifesto']))
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
                       how='inner',     # TODO: investigate if this should be changed to left
                       left_index=True,
                       right_index=True)

    #factorize features with small cardinality
    # Path(new_features_dir).mkdir(exist_ok=True, parents=True)
    # with get_dask_compute_environment(config) as client:
    #     for col in config["low_cardinality_rehash_features"]:
    #         tmp_col = f'{col}_encode'
    #         fut_tmp = ddf[col].unique()
    #         fut_tmp = client.persist(fut_tmp)
    #         if verbosity > 0:
    #             progress(fut_tmp)
    #         tmp = fut_tmp.compute()
    #         tmp = tmp.to_frame().reset_index()
    #         tmp.columns = [i if i!="index" else tmp_col for i in tmp.columns]
    #         ddf = ddf.merge(tmp, on=col, how='left')
    #         ddf[tmp_col] = ddf[tmp_col].astype('uint8')
    #         mapping_output = join(new_features_dir, f"{col}_mapping.csv")
    #         if verbosity >= 1: print(f"Outputing mapping for {col} to {mapping_output}")
    #         tmp[[tmp_col, col]].to_csv(mapping_output)


    # then add the TEs as per configuration
    for te_feature, te_targets in config['TE_features'].items():
        for te_target in te_targets:
            if train_set_mode:
                ddf, cnm = TE_dataframe_dask(ddf, te_feature, te_target)       # already joins in the function
                cnm.to_parquet()
            else:
                cnm = dd.read_parquet()
                ddf, cnm = TE_dataframe_dask(ddf, te_feature, te_target)  # already joins in the function


    new_feature_columns = [col for col in ddf.columns if col not in original_cols]
    if verbosity >= 1:
        print("The following preprocessed columns are dumped: ", new_feature_columns)
    delayed_dump = ddf[new_feature_columns].to_parquet(new_features_dir, compute=False)

    if verbosity >= 1:
        print("Feature generation ready, associated delayed objects: {}\n\n{}.".format(ddf, delayed_dump))

    return ddf, delayed_dump

