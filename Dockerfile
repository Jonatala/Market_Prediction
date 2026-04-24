# Lightweight python image
FROM python:3.10-slim

# set working directory
WORKDIR /app

#copy project files
COPY . .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Default command: run pipeline
CMD ["python", "src/pipeline/run_pipeline.py"]