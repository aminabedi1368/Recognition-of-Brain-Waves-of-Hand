FROM python:3.7.1

RUN apt-get update && apt-get install -y python-pip

RUN mkdir /usr/src/app

WORKDIR /usr/src/app

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD python index.py
