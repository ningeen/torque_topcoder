FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

RUN mkdir /code
COPY ["./predict_test.py", "./cosine_annearing_with_warmup.py", "./mobilenetv3.py", "./pytorch_dataset.py", "./pytorch_model.py", "./read_and_get_mel.py", "./proj_config.yaml", "./requirements.txt", "./train.py", "./train.sh", "./pred.sh", "/code/"]
COPY ./model/mixup/* /code/weights/
COPY ./model/origin_2ch/* /code/weights/
COPY ./model/1ch_new_augs/* /code/weights/
RUN pip install -r /code/requirements.txt && apt update && apt-get -y install libsndfile1 && chmod +x /code/train.sh && chmod +x /code/pred.sh