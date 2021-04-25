from utils.preprocess import preprocess, get_dask_compute_environment, load_all_preprocessed_data
from dask.distributed import progress


if __name__=='__main__':
    _, delayed_save = preprocess()

    with get_dask_compute_environment() as client:
        f = client.persist(delayed_save)
        progress(f)
        f.compute()

    # TODO: add conditional probability computation with ..._per_config ?
 
