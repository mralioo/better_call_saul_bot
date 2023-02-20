FROM rasa/rasa:2.8.3-full

USER root
RUN apt update && \
    apt install -y git \
        make \
        wget

RUN pip install black \
    ipywidgets \
    jupyterlab \
    transformers==4.9.2