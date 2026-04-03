FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p trajectories training_trajectories

ENV PYTHONUNBUFFERED=1
ENV HF_TOKEN=""
ENV OPENROUTER_API_KEY=""

EXPOSE 7860

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860"]
