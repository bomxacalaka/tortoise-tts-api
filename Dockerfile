FROM drbom/tts:0.8


COPY . /app
ADD . /app
# --login option used to source bashrc (thus activating conda env) at every RUN statement
SHELL ["/bin/bash", "--login", "-c"]

# pip install gevent
RUN conda activate tortoise \
    && pip install gevent
    


COPY serve.sh ./
RUN chmod +x serve.sh
SHELL ["/bin/bash", "-c"]
WORKDIR /app



# Activate the environment toroise and run the server.sh script
ENTRYPOINT ["./serve.sh"]


# docker rm ttscontainer & docker image rmi drbom/tts2:0.0 & docker build . -t drbom/tts2:0.0

# docker run --name ttscontainer -p 5000:5000 --gpus all -e TORTOISE_MODELS_DIR=/models -v /mnt/user/data/tortoise_tts/models:/models -v /mnt/user/data/tortoise_tts/results:/results -v /mnt/user/data/.cache/huggingface:/root/.cache/huggingface -v /root:/work -it europe-west2-docker.pkg.dev/lango2lang-5d4c3/app-containers-repo/drbom/tts2:0.0

# docker image rmi drbom/tts2:0.0

# docker build . -t drbom/tts2:0.0

# conda activate tortoise

# gunicorn --chdir tortoise  -b :5000 flask-api:app

# docker rm ttscontainer


# Full deployment commands:

# docker build -t europe-west2-docker.pkg.dev/lango2lang-5d4c3/app-containers-repo/drbom/tts2:0.0 .

# docker push europe-west2-docker.pkg.dev/lango2lang-5d4c3/app-containers-repo/drbom/tts2:0.0

# gcloud container clusters create tts-api --zone europe-west1-b --num-nodes 1

# kubectl create deployment tts-api-deployment --image=europe-west2-docker.pkg.dev/lango2lang-5d4c3/app-containers-repo/drbom/tts2

# kubectl expose deployment tts-api-deployment --type=LoadBalancer --port 80 --target-port 5000

# kubectl get all 