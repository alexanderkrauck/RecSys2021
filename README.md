# AI-Warriors
AI-Warriors

# environment setup ( until we get to the docker image ) 

conda create --name twitterchallenge python=3.9

conda activate twitterchallenge

conda install -c conda-forge python-lzo

conda install -c conda-forge dask

conda install -c conda-forge jupyterlab 

conda install -c conda-forge fastparquet 

# for bert tokenizer

conda install -c conda-forge transformers

conda install -c conda-forge ipywidgets

# only valid if you have linux and 30x gpu

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
