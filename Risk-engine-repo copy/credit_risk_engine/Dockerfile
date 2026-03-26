# it ensures the system runs identically everywhere:
    # developer machine 
    # NBFC infrastructure
    # cloud server
    # regulator replay environment
docker build -t pd-engine:1.0 . 
# a regulator replaying the model must obtain identical PD values.
FROM python:3.10.13-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY configs/ ./configs
COPY models/ ./models

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "credit_risk_engine.scoring_service"]  