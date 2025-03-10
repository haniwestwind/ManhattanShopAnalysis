FROM python:3.10-buster

# Install system dependencies needed for geopandas
RUN apt-get update && apt-get install -y \
    libspatialindex-dev \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev
RUN apt-get update && apt-get install -y bash  # Install bash

# Install Python dependencies
COPY requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt

WORKDIR /projects


# Set environment variables
ENV MONGO_URI="mongodb://localhost:27017/yelp_data"

CMD ["/bin/bash"]  