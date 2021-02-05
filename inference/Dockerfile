FROM python:3

RUN mkdir /code
COPY [    "./predict_test.py",    "./cosine_annearing_with_warmup.py",    "./mobilenetv3.py",    "./pytorch_dataset.py",    "./pytorch_model.py",    "./read_and_get_mel.py",    "./proj_config.yaml",    "./requirements.txt",    "/code"]
COPY ./pretrained/work_one_cosine_fold0.pt /tmp/weights.pt
RUN pip install -r /code/requirements.txt && chmod +x /code/train.sh && chmod +x /code/pred.sh

