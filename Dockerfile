# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .

ARG tag=2.9.1-cuda12.6-cudnn9-runtime

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM pytorch/pytorch:${tag}

LABEL maintainer='Josep Oliver-Sanso, Jesus Soriano-Gonzalez'
LABEL version='0.0.1'

# Install Ubuntu packages
# - gcc is needed in Pytorch images because deepaas installation might break otherwise (see docs) (it is already installed in tensorflow images)
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
        curl \
        nano \
        psmisc \
    && rm -rf /var/lib/apt/lists/*

    # Set LANG environment
ENV LANG=C.UTF-8

# Set the working directory
WORKDIR /srv

# Initialization scripts
# deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/ai4os/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# TODO: Install from gitclone in a future
COPY . /srv/socib-shoreline-extraction
WORKDIR /srv/socib-shoreline-extraction

RUN pip3 install --no-cache-dir -e .
RUN pip3 install --no-cache-dir tox

# Open ports: DEEPaaS (5000)
EXPOSE 5000

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000", "--config-file", "deepaas.conf"]

# Healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=5 \
  CMD curl --fail http://localhost:5000/v2  