# Use official Python 3.10 image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for OpenCV and others)
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev gcc \
    && apt-get clean

# Copy project files to the container
COPY . .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port if using Flask
EXPOSE 5000

# Run your app (change app.py to your entry file if different)
CMD ["python", "app.py"]
