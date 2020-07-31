FROM python:3.8.3
WORKDIR /breastcancerapp
ADD . /breastcancerapp
RUN pip install -r requirements.txt
CMD ["python","app.py"]