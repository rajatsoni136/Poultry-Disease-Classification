# 1. Start with the slim image
FROM python:3.8-slim

# 2. Install "Build Tools" required for compiling h5py and others
#    We added: pkg-config, libhdf5-dev, gcc, and build-essential
RUN apt-get update -y && \
    apt-get install -y awscli pkg-config libhdf5-dev gcc build-essential

WORKDIR /app

# 3. Copy requirements
COPY requirements.txt /app/

# 4. Install Python packages (Now it has the tools to build them!)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy app code
COPY . /app

# 6. Open the port
EXPOSE 8080

# 7. Run the app
CMD ["python3", "app.py"]