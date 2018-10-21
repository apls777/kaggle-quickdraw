FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /root

# install gsutil
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-`lsb_release -c -s` main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y google-cloud-sdk python2.7 vim

ENV CLOUDSDK_PYTHON="/usr/bin/python2.7"

# install vim and tzdata
RUN apt-get -y install vim tzdata

# change timeone
ENV TZ="Europe/Dublin"

# install awscli
RUN pip3 install awscli

ENV PYTHONPATH="/workspace/project"

# for TensorFlow to read TFRecords from S3
ENV AWS_REGION=us-east-2
