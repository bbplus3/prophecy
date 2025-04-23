# Use a lightweight Python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install \
    streamlit \
    pandas \
    numpy \
    nltk \
    scikit-learn \
    joblib 

# Download NLTK stopwords (optional if used)
# RUN python -m nltk.downloader stopwords

# Expose Streamlit's default port
EXPOSE 8080

# Streamlit command to run app
CMD ["streamlit", "run", "fulfilled.py", "--server.port=8080", "--server.address=0.0.0.0"]
