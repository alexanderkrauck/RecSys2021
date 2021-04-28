import yaml
from typing import Union, Dict
import os
from constants import FILE_DIR


def load_config(file_path: Union[os.PathLike, str] = None) -> dict:
    '''

    Parameters
    ----------
    file_path: string or path or none
        If None, then the default configuration is loaded.
    Returns
    -------
        Configuration as a nested dictionary.
    '''
    if not file_path:
        file_path = os.join(FILE_DIR, "default_config.yaml")
    with open(file_path) as f:
        return yaml.load(f, Loader=yaml.CLoader)


def load_compute_config(file_path: Union[os.PathLike, str] = None, cfg: Dict = None) -> dict:
    if not cfg:
        cfg = load_config(file_path)
    retcfg = cfg['compute']
    retcfg['verbose'] = cfg['verbose']
    return retcfg


def load_mop_config(file_path: Union[os.PathLike, str] = None, cfg: Dict = None) -> dict:
    if not cfg:
        cfg = load_config(file_path)
    retcfg = cfg['mop']
    retcfg['verbose'] = cfg['verbose']
    return retcfg


def load_feature_config(file_path: Union[os.PathLike, str] = None, cfg: Dict = None) -> dict:
    if not cfg:
        cfg = load_config(file_path)
    retcfg = cfg['mop']
    retcfg['verbose'] = cfg['verbose']
    return retcfg


def load_manifest(file_path: Union[os.PathLike, str]) -> dict:
    return load_config(file_path)


def dump_manifest(file_path: Union[os.PathLike, str], manifest: Dict) -> dict:
    with open(file_path, "wt") as f:
        yaml.dump(manifest, f)
