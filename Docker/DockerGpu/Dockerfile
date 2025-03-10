# Use a TensorFlow GPU base image
FROM tensorflow/tensorflow:2.15.0-gpu 

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libspatialindex-dev \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    bash

# Install Python dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /projects


# Set environment variables
ENV MONGO_URI="mongodb://localhost:27017/yelp_data"

# Verify GPU availability
RUN python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

CMD ["/bin/bash"]