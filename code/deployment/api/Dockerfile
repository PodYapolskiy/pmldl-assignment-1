FROM python:3.12-slim

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["fastapi", "run", "/code/deployment/api/api.py", "--port", "8000"]
