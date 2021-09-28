FROM python:3

ENV PYTHONUNBUFFERED=1
COPY . /anomaly_detection_project
WORKDIR /anomaly_detection_project
RUN pip install -r requirements.txt

CMD ["python3", "main.py"]