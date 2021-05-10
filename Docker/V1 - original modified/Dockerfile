FROM gcr.io/google-appengine/python

RUN virtualenv /env -p python3.7
# Setting these environment variables are the same as running
# source /env/bin/activate.
ENV VIRTUAL_ENV /env
ENV PATH /env/bin:$PATH

RUN pip install torch==1.7.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt
RUN python -m pip install "dask[complete]" 

ADD . /app
WORKDIR /app