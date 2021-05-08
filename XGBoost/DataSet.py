from dask import dataframe as dd
from torch.utils.data import Dataset

class XGSet(Dataset):
    """General Dataset to use it as an intermediate structure.
    Key Idea, more elaborate datasets in getitem method are calling this one, receive preprocessed data
    and label and process them more if needed"""

    def __init__(self, ddf,features:list, clazz:str):
        """Initialise ddf and features vs label column
        :parameter ddf DaskDataframe
        :parameter features list of strings, names of feat columns
        :parameter clazz: str name of target column
        """
        self.ddf = ddf
        self.features = features
        self.clazz = clazz
        #turn to int binary
        ddf[self.clazz] = ddf[self.clazz].astype(int)

    def __len__(self):
        return self.ddf.shape[0].compute()


    def form_subset(self,tuple):
        """Forms a subset based on parameters
        :parameter tuple (start:int, end:int) specifies arranged indicies we will work with
        Since our data is sequential, and ddf should be already sorted it is a valid approach"""
        lst = list(range(tuple[0],tuple[1]))

        #form subset
        features_batch = self.ddf.loc[lst][self.features].values.compute_chunk_sizes()
        labels_batch = self.ddf.loc[lst][self.clazz].values.compute_chunk_sizes()

        return (features_batch,labels_batch)