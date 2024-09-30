# PMLDL MLOps Assigment 

## Clone and install dependencies

```bash
git clone https://github.com/PodYapolskiy/pmldl-assignment-1.git
cd pmldl-assignment-1
```

```bash
poetry shell
```

```bash
poetry install --no-root
```

## Run

#### Development

Setup backend

```bash
fastapi dev code/deployment/api/api.py
```

Right after setup ui via streamlit

```bash
streamlit run code/deployment/app/app.py
```

#### Production

Use docker compose

```bash
docker compose up -d
```
