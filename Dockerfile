FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command to run the application
CMD ["python3", "main.py"]
