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

# Install socib-shoreline-extraction (main)
RUN git clone --depth 1 -b main https://github.com/ai4os-hub/socib-shoreline-extraction && \
    cd socib-shoreline-extraction && \
    pip3 install --no-cache-dir -e .

# Download pretrained models (oblique and rectified)
RUN mkdir -p /srv/socib-shoreline-extraction/models && \
    curl -L https://github.com/ai4os-hub/socib-shoreline-extraction/releases/download/v.0.1.0/oblique_best_model.pth \
    --output /srv/socib-shoreline-extraction/models/oblique_best_model.pth && \
    curl -L https://github.com/ai4os-hub/socib-shoreline-extraction/releases/download/v.0.1.0/rectified_best_model.pth \
    --output /srv/socib-shoreline-extraction/models/rectified_best_model.pth

# Set working directory
WORKDIR /srv/socib-shoreline-extraction

# Open ports: DEEPaaS (5000)
EXPOSE 5000

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000", "--config-file", "deepaas.conf"]