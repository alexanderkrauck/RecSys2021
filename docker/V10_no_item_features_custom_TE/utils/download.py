"""
Utility functions for downloading files from the RecSys2020 Servers

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "08-03-2021"


import os
from os.path import join
from pathlib import Path
from typing import Union, Tuple
from datetime import datetime

def download_data(url_file="training_urls.txt", output_dir="downloaded_data", uncompress=True,
                  delete_compressed=True, verbose=1, only_indices=None):
    """Download data from specified file (in the format from the website) to the directory specified    

    Parameters
    ----------
    url_file : str
        The location of the file containing the urls (each line is one url)
    output_dir : str
        The location of the directory where the downloaded data should be stored (will be created if it does not exist)
    uncompress : bool
        If true then the data will be extracted
    delete_compressed : bool
        If true then the compressed version of the data will be removed after extraction. (Requires "uncrompress" to be true)
    verbose : int
        Decides level of verboseness. (Max 1, Min 0)
    only_indices : list
        for partial downloads
    """ 

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(url_file, "r") as f:
        lines = f.readlines()   

    idx = 0
    for line in lines:
            if "index" not in line:
                download_filename = join(output_dir, f"part-{idx:05}.lzo")
                if verbose > 0: print(
                    f"Downloading {line[:30]} to {download_filename}... ", end="")
                os.system(f"wget -O {download_filename} -x \"{line}\"")
                if verbose > 0: print("done")   

                if uncompress:
                    uncompress_filename = join(
                        output_dir, f"part-{idx:05}.tsv")
                    if verbose > 0: print(
                        f"Uncompressing {download_filename}... ", end="")
                    os.system(
                        f"lzop -o {uncompress_filename} -d {download_filename}")
                    if verbose > 0: print("done")   

                    if delete_compressed:
                        if verbose > 0: print(
                            f"Deleting {download_filename}... ", end="")
                        os.remove(download_filename)
                        if verbose > 0: print("done")
                idx += 1


def fix_file_naming(file_dir: str):
    """Remove the unwanted part of the filename

    Parameters
    ----------
    file_dir : str
        The directory where to do the operation
    """
    for file in os.listdir(file_dir):
        if "@" in file:  # Windows
            old_file = os.path.join(file_dir, file)
            new_file = os.path.join(file_dir, file.split('@')[0])
            os.rename(old_file, new_file)
        if "?" in file:  # GNU/Linux
            old_file = os.path.join(file_dir, file)
            new_file = os.path.join(file_dir, file.split('?')[0])
            os.rename(old_file, new_file)


def decompress_lzo_file(file_dir: str, target_dir: str, delete_compressed=True, overwrite=False, verbose=1):
    """Decompress lzo files in the given directory

    Parameters
    ----------
    file_dir : str
        The directory that contains the lzo files
    target_dir : str
        The directory to extract to
    delete_compressed : bool
        If true then the compressed version of the data will be removed after extraction
    overwrite : bool
        If true then this will overwrite existing unpacked files
    verbose : int
        Decides level of verboseness. (Max 1, Min 0)
    """

    files = os.listdir(file_dir)
    for file in files:
        if file.endswith(".lzo"):
            compressed_file = join(file_dir, f"{file}")
            uncompress_filename = join(target_dir, f"{file[:-4]}.tsv")

            if not (os.path.exists(uncompress_filename) or overwrite):
                if verbose > 0: print(f"Uncompressing {compressed_file}... ", end="")
                os.system(f"lzop -o {uncompress_filename} -d {compressed_file}")
                if verbose > 0: print("done")

                if delete_compressed:
                    if verbose > 0: print(f"Deleting {compressed_file}... ", end="")
                    os.remove(compressed_file)
                    if verbose > 0: print("done")


def readable_time_now() -> str:
    '''

    Returns
    -------
        current time in a readable format
    '''
    return datetime.now().strftime('__%m_%d_%H_%M')


def shelf_directory(dir_path: Union[os.PathLike, str]) -> str:
    '''

    Parameters
    ----------
    dir_path: path/string
        the directory to be renamed
    Returns
    -------
        the new name with readable timestamp that the directory was renamed to
    '''
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        new_path = str(dir_path)+readable_time_now()
        os.rename(dir_path, new_path)
    else:
        new_path = None
    os.mkdir(dir_path)

    return new_path


def ensure_dir_exists(dir_path: Union[os.PathLike, str]) -> bool:
    '''

    Parameters
    ----------
    dir_path
        directory path
    Returns
    -------
        True if the directory already existed before
    '''
    if os.path.exists(dir_path):
        return True
    else:
        os.mkdir(dir_path)
        return False