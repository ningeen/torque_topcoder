FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN mkdir /code && mkdir /input && mkdir /pretrained
COPY ["./proj_config.yaml", "./logging.conf.yml", "./requirements.txt", "./inference/train.sh", "./inference/pred.sh", "/code/"]
COPY /src/* /code/
COPY /model/bootstrap/* /code/weights/
COPY /model/bootstrap_2ch/* /code/weights/
COPY /pretrained/mobilenetv3-large-1cd25616.pth /pretrained/mobilenetv3-large-1cd25616.pth
RUN pip install -r /code/requirements.txt && apt update && apt-get -y install libsndfile1 && chmod +x /code/train.sh && chmod +x /code/pred.sh