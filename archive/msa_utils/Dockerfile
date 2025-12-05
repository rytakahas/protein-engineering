# msa_utils/Dockerfile

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget unzip perl python2 python-is-python2 \
    build-essential zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN wget https://zhanggroup.org/DeepMSA/package.zip && \
    unzip package.zip && \
    rm package.zip

ENV PATH="/opt/hhsuite2/bin:/opt/DeepMSA_v1.6/:${PATH}"

RUN chmod +x /opt/hhsuite2/bin/*

ENTRYPOINT ["/bin/bash"]

