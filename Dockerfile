FROM nvidia/cuda:base

WORKDIR /usr/src/app

RUN apt-get update && apt-get install -y python3 python3-pip

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN curl -O https://raw.githubusercontent.com/huggingface/deep-rl-class/main/notebooks/unit1/requirements-unit1.txt && pip install -r requirements-unit1.txt

RUN pip install -r requirements-unit1.txt
   




