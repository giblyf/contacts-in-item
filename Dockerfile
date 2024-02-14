FROM --platform=linux/x86_64 nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

USER root

RUN apt-get update && \
	apt-get install -y curl python3.8 python3.8-distutils wget && \
	ln -s /usr/bin/python3.8 /usr/bin/python && \
	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py && \
    python -m pip install -U pip==20.3.3

ENV PROJECT_ROOT /app

ENV DATA_ROOT /data
ENV TEST_DATA_ROOT /test_data

RUN mkdir $PROJECT_ROOT $DATA_ROOT

COPY . $PROJECT_ROOT

WORKDIR $PROJECT_ROOT

RUN pip install -r requirements.txt

RUN cd $PROJECT_ROOT/data && chmod +x get_train_data.sh && ./get_train_data.sh \
    && chmod +x train.csv && chmod +x get_preprocessor.sh && ./get_preprocessor.sh

RUN python -m nltk.downloader punkt stopwords

RUN cp -r $PROJECT_ROOT/data/* $DATA_ROOT && rm -rf $PROJECT_ROOT/data

CMD python lib/run.py --debug