FROM python:3.10-bullseye

# Set work directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libffi-dev \
    gcc \
    g++ \
    curl \
    && apt-get clean

# Pre-install numpy (required for building scikit-surprise)
RUN pip install --upgrade pip
RUN pip install numpy==1.23.5

# Install Python build dependencies
RUN pip install cython==0.29.36

# Copy requirements and install all packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy app code
COPY . .

# Expose port 5000 (matches Gunicorn and Azure)
EXPOSE 5000

# Run the app on port 5000
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
