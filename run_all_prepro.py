from utils.preprocess import preprocess
import utils.config as config
from dask.distributed import progress, wait
import utils.compute_and_front as compute





if __name__=='__main__':


    cfg = config.load_config("mach2_train_config.yaml")

    ddf, delayed_dump, other_delayed = preprocess.preprocess(cfg)

    compconf = cfg['compute']
    compconf['verbose'] = cfg['verbose']

    with compute.get_dask_compute_environment(compconf) as client:
        f = client.persist([delayed_dump, *other_delayed])
        progress(f)
        wait(f)

    

    print("Training Data Done!")

    cfg = config.load_config("mach2_validation_config.yaml")

    ddf, delayed_dump, other_delayed = preprocess.preprocess(cfg)

    compconf = cfg['compute']
    compconf['verbose'] = cfg['verbose']

    with compute.get_dask_compute_environment(compconf) as client:
        f = client.persist(delayed_dump)
        progress(f)
        f.compute()

    print("Validation Data Done!")
