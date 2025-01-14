FROM python:3.7.4-slim-stretch

WORKDIR /app
COPY . /app
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["flask", "run", "-h", "0.0.0.0"]
