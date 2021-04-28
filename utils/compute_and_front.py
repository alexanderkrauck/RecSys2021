import os
from typing import Union, Tuple, Dict

from config import load_feature_config, load_compute_config, load_mop_config

from dask import dataframe as dd
from features import conditional_probabilities
from dask.distributed import Client, progress

from download import decompress_lzo_file
from constants import COMP_DIR, all_columns, all_features, dtypes_of_features, converters_for_the_original_dataset

def conditional_probabilities_as_per_config(df: dd.DataFrame,
                                            feat_config: dict = None):
    '''

    '''
    if not feat_config:
        feat_config = load_feature_config()

    features = feat_config['marginal_prob_columns']['features']
    labels = feat_config['marginal_prob_columns']['per_labels']

    with get_dask_compute_environment(feat_config) as client:
        delayed_series = conditional_probabilities(df, features, labels)
        delayed_op = dd.to_csv(delayed_series, compute=False)
        f = client.persist(delayed_op)
        if feat_config['verbose'] > 0:
            progress(f)
        f.compute()

    return



def load_all_preprocessed_data(
        new_features: bool = False,
        old_features: bool = True,
        mop_config: Dict = None
    ) -> dd.DataFrame:
    '''

    Parameters
    ----------
    mop_config : object
    new_features bool
        whether to load the synthetic features
    old_features : bool
        whether to load the original features
    Returns
    -------
        A lazy dataframe that has the original preprocessed features and new computed features loaded
    '''
    if not mop_config:
        mop_config = load_mop_config()

    prepro_dir = os.path.join(COMP_DIR, mop_config['preprocessed_dir'])
    new_features_dir = os.path.join(COMP_DIR, mop_config['feature_dir'])

    if new_features and old_features:
        original_df = dd.read_parquet(prepro_dir, index="idx")
        extra_features_df = dd.read_parquet(new_features_dir, index="idx")
        return dd.merge(original_df, extra_features_df, how='inner', left_index=True, right_index=True)  # on index, so fast
    elif new_features:
        return dd.read_parquet(prepro_dir, index="idx")
    elif old_features:
        return dd.read_parquet(new_features_dir, index="idx")
    else:
        raise Exception('Life is vane, you send me to get no data.')


def get_dask_compute_environment(comp_config: dict = None) -> Client:
    '''

    Parameters
    ----------
    comp_config
        computational configuration
    Returns
    -------
        a client object that is already set as default in the current context
    '''
    if not comp_config:
        comp_config = load_compute_config()

    verbosity = comp_config['verbose']
    client_n_workers = comp_config['n_workers']
    client_n_threads = comp_config['n_threads_per_worker']
    client_memlim = comp_config['mem_lim']

    # initialize a client according to the configuration and make it local
    client = Client(memory_limit=client_memlim,
                    n_workers=client_n_workers,
                    threads_per_worker=client_n_threads,
                    processes=True)
    client.as_current()     # this assigns all the operations implicitly to this client, I believe.

    if verbosity >= 1:
        print("Compute environment established: {}".format(client))

    return client


def uncompress_and_parquetize(comp_dir: str = COMP_DIR, config: dict = None):

    mop_config = load_mop_config(config)
    compute_config = load_compute_config(config)

    #Load default config if not specified and extract required parameters
    verbosity = mop_config['verbose']

    # unpack some of the config variables
    train_set_mode = mop_config['train_set']
    data_source = mop_config['load_from']

    #Set default directories if not specified (based on root dir)
    comp_dir = os.path.join(comp_dir, mop_config['compressed_dir'])
    uncomp_dir = os.path.join(comp_dir, mop_config['uncompressed_dir'])
    prepro_dir = os.path.join(comp_dir, mop_config['preprocessed_dir'])

    if data_source == 'comp':
        # decompress lzo files to uncomp directory
        decompress_lzo_file(comp_dir, uncomp_dir, delete_compressed=False, overwrite=False, verbose=verbosity)

    # start with reading the files lazily and locally, assume they are uncompressed
    unpacked_files = [os.path.join(uncomp_dir, f)
                      for f in os.listdir(uncomp_dir) if os.path.isfile(os.path.join(uncomp_dir, f))]
    if verbosity >= 2:
        print(unpacked_files)

    if train_set_mode:
        ddf = dd.read_csv(unpacked_files, sep='\x01', header=None, names=all_columns, blocksize=compute_config['chunksize'],
                          dtype=dtypes_of_features,
                          converters=converters_for_the_original_dataset
                          )  # TODO: add dtypes from above and converters
    else:
        ddf = dd.read_csv(unpacked_files, sep='\x01', header=None, names=all_features, blocksize=compute_config['chunksize'],
                          dtype=dtypes_of_features,
                          converters=converters_for_the_original_dataset
                          )  # TODO: add dtypes from above and converters

    # Add unique id for indexing
    ddf["idx"] = 1
    ddf["idx"] = ddf["idx"].cumsum()

    # deal with the NAs in the source data
    ddf['reply'] = ddf['reply'].fillna(0)
    ddf['retweet'] = ddf['retweet'].fillna(0)
    ddf['retweet_comment'] = ddf['retweet_comment'].fillna(0)
    ddf['like'] = ddf['like'].fillna(0)

    if verbosity > 0: print("Outputting preprocessed files")
    with get_dask_compute_environment(compute_config) as client:
        # now drop the resulting dataframe to the location where we can find it again
        futures_dump = ddf.to_parquet(prepro_dir, compute=False)
        # works under assumption that the local machine shares the file system with the dask client
        f = client.persist(futures_dump)
        if verbosity > 0:
            progress(f)
        f.compute()

    if verbosity >= 1:
        print("The uncompressed dataset is saved to the disk.")

    # free up the memory
    del ddf, f
