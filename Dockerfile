FROM python:3.8.3
WORKDIR /breastcancerapp
ADD . /breastcancerapp
RUN pip install -r requirements.txt
ENV PORT 8080
CMD ["gunicorn", "app:app"]