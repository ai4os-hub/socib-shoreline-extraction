# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone (default: master, another option: test)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# Be Aware! For the Jenkins CI/CD pipeline,
# input args are defined inside the JenkinsConstants.groovy, not here!

ARG tag=2.9.1-cuda12.6-cudnn9-runtime

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM pytorch/pytorch:${tag}

LABEL maintainer='Josep Oliver-Sanso, Jesus Soriano-Gonzalez'
LABEL version='0.0.1'
# A demo application to test (eg. DEEPaaS testing). Does not contain any AI code.

# What user branch to clone [!]
ARG branch=main

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
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# Disable FLAAT authentication by default
ENV DISABLE_AUTHENTICATION_AND_ASSUME_AUTHENTICATED_USER yes

# Initialization scripts
# deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/ai4os/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# Necessary for the Jupyter Lab terminal
ENV SHELL /bin/bash

# TODO: Install from gitclone in a future
COPY . /srv/socib-shoreline-extraction
WORKDIR /srv/socib-shoreline-extraction

RUN pip3 install --no-cache-dir -e .
RUN pip3 install --no-cache-dir tox

# Open ports: DEEPaaS (5000), Monitoring (6006), Jupyter (8888)
EXPOSE 5000 6006 8888

# Launch deepaas
CMD ["deepaas-run", "--listen-ip", "0.0.0.0", "--listen-port", "5000", "--config-file", "deepaas.conf"]

# Healthcheck
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=5 \
  CMD curl --fail http://localhost:5000/v2  