FROM python:3.8.3
WORKDIR /breastcancerapp1
ADD . /breastcancerapp1
RUN pip install -r requirements.txt
CMD ["python","app.py"]