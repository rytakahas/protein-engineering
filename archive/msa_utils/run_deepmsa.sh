#!/bin/bash
# msa_utils/run_deepmsa.sh

docker build -t deepmsa ./msa_utils
docker run -it --rm \
  -v $(pwd)/data:/workspace/data \
  deepmsa

