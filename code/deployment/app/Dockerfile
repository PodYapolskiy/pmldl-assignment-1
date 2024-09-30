FROM python:3.12-slim

RUN mkdir /streamlit

COPY ./requirements.txt /streamlit/requirements.txt

WORKDIR /streamlit

RUN pip install -r requirements.txt

COPY ./code/deployment/app /streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
