# Use official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disk and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies including Tesseract OCR and PyMuPDF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    libtesseract-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy remaining application code
COPY . .

# Generate sample resumes inside container
RUN python generate_samples.py

# Expose Streamlit port
EXPOSE 8501

# Healthcheck to monitor app status
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Command to run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
