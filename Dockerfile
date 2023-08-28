FROM --platform=linux/amd64 amazonlinux:2 AS base
#MAINTAINER subhomoysikdar sikdarsubhomoy@gmail.com

RUN yum install -y python3

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install numpy && \
    python3 -m pip install pandas && \
    python3 -m pip install tqdm && \
    python3 -m pip install joblib && \
    python3 -m pip install xgboost && \
    python3 -m pip install scikit-learn && \
    python3 -m pip install scipy && \
    python3 -m pip install ipywidgets && \
    python3 -m pip install s3fs && \
    python3 -m pip install tsfresh && \
	python3 -m pip install pyarrow \
    venv-pack==0.2.0

RUN mkdir /output && venv-pack -o /output/pyspark_env.tar.gz


FROM scratch AS export
COPY --from=base /output/pyspark_env.tar.gz /
