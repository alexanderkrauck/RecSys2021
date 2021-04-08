"""
Utility functions for downloading files from the RecSys2020 Servers

"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "08-03-2021"


import os
from os.path import join
from pathlib import Path


def download_data(url_file="training_urls.txt", output_dir="downloaded_data", uncompress=True, delete_compressed=True, verbose=1):
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
        The directory where to do the opteration
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

def decompress_lzo_file(file_dir: str, delete_compressed=True, verbose=1):
    """Decompress lzo files in the given directory

    Parameters
    ----------
    file_dir : str
        The directory where to do the opteration
    delete_compressed : bool
        If true then the compressed version of the data will be removed after extraction
    verbose : int
        Decides level of verboseness. (Max 1, Min 0)
    """

    files = os.listdir(file_dir)
    for file in files:
        if file.endswith(".lzo"):
            compressed_file = join(file_dir, f"{file}")
            uncompress_filename = join(file_dir, f"{file[:-4]}.tsv")
            if verbose > 0: print(f"Uncompressing {compressed_file}... ", end="")
            os.system(f"lzop -o {uncompress_filename} -d {compressed_file}")
            if verbose > 0: print("done")   

            if delete_compressed:
                if verbose > 0: print(f"Deleting {compressed_file}... ", end="")
                os.remove(compressed_file)
                if verbose > 0: print("done")