# Team JKU-AIWarriors in the ACM Recommender Systems Challenge 2021: Lightweight XGBoost Recommendation Approach Leveraging User Features
by Alexander Krauck, David Penz and Markus Schedl

This repository contains our 10-th place approach to the ACM Recommender Systems Challenge 2021 (http://www.recsyschallenge.com/2021/). Our proposed model relies on features that we compute from user engagement counts. These counts are used to create compact user-specific features, which enables our model to make predictions swiftly. We adopt a simple XGB classifier, trained on a subset of the training data. To regularize during training, adding Gauss-distributed noise and randomly masking users helps to avoid overfitting. 

## How to reproduce our results
The utils folder contains all the relevant code for reproducing the model. Basically 2 steps are required:
 1. Use the functions from utils/user_based.py to create the "user index". I.e. extract user-information from the whole dataset.
 2. Create a RecSys2021TSVDataLoader (in utils/dataloader.py) and pass the dataloader to a RecSysXGB1 (in utils/model.py) using the fit function.

Given the right choice of hyperparameters this will successfully reproduce our submission model.

### Jupyter Notebooks
The jupyter notebooks are simply our way of executing the code in the utils as opposed to some command line approach or similar. Thus the order of the notebooks simply represents how we progressed thru the challenge. Some code in the earier notebooks might not be working anymore. They do not represent a working pipeline.
