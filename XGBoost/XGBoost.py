import xgboost as xgb
import numpy as np
import dask
import os.path as path
from sklearn.metrics import average_precision_score, log_loss


class XGBoost():
    def __init__(self,conf:dict):
        """:parameter conf specifies hyperparameters of XGBoost"""
        self.parameters = conf
        self.model = None

    def fit(self,X,y,model_in,models_dir,i,client):
        """Function which takes a df or ddf uses fit saves model.
        :parameter X nd array of features
        :parameter y 1d array of labels
        :parameter model_in Path to input model, needed for test, eval, dump tasks.
        If it is specified in training, XGBoost will continue training from the input model.
        :parameter models_dir:str, path to directory with saved models
        :parameter i round of fitting
        :parameter client is a dask client"""
        dmatrix = xgb.dask.DaskDMatrix(client,X,y)
        self.model = xgb.dask.train(client,self.parameters,dmatrix,xgb_model=model_in,feval=self.rce)['booster']
        self.model.save_model(path.abspath((path.join(models_dir,f'model_{i}.model'))))
        return path.abspath((path.join(models_dir,f'model_{i}.model')))

    def predict(self,X,y,client,model):
        dmatrix = xgb.dask.DaskDMatrix(client, X)#label=y ???
        booster = xgb.Booster()
        booster.load_model(model)
        pred = xgb.dask.predict(client,booster,dmatrix)
        return pred

    def rce(self,pred: np.ndarray, dtrain: xgb.DMatrix):
        """Function is being used by xgboost train as optimisation objective
        :parameter pred predicted values
        :parameter dtrain Dmatrix, consists of X and y data.
        :return tuple (str, float), where string is a name of eval function"""

        y = dtrain.get_label()
        return 'RCE', self.compute_rce(pred,y)

    def calculate_ctr(sefl,gt):
        """ Copied from RecSys."""
        positive = len([x for x in gt if x == 1])
        ctr = positive / float(len(gt))
        return ctr

    def compute_rce(self, pred, gt):
        """ Copied from RecSys."""
        cross_entropy = log_loss(gt, pred)
        data_ctr = self.calculate_ctr(gt)
        strawman_cross_entropy = log_loss(gt, [data_ctr for _ in range(len(gt))])
        return (1.0 - cross_entropy / strawman_cross_entropy) * 100.0

    def evaluate(self, pred, true):
        """Function is used outside of xgboost.train for evaluation on validation and test set."""
        pred = pred.compute()
        true = true.compute()
        ap = average_precision_score(true, pred)
        rce = self.compute_rce(pred,true)
        del pred, true
        return ap, rce