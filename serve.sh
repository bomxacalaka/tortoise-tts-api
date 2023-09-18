#!/bin/bash
# run with gunicorn (http://docs.gunicorn.org/en/stable/run.html#gunicorn)

# . /opt/conda/etc/profile.d/conda.sh
# /root/miniconda
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate pytorch

# # Temporarily disable strict mode and activate conda:
set +euo pipefail
source /root/miniconda/etc/profile.d/conda.sh
conda activate tortoise

# Re-enable strict mode:
set -euo pipefail

# mv models/* /models/ \
#     && rm -rf models

# Move models to /models if they exist
if [ -d "models" ]; then
    mv models/* /models/ \
        && rm -rf models
fi

exec gunicorn --chdir tortoise  -b :5000 --worker-class gevent flask-api:app

# echo "Starting gunicorn"