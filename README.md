### README

## How to set up

1. Set up the docker.

docker build -t business_predictor_env .

2. Run the docker container

docker run --network="host" -v $(pwd):/projects -it business_predictor_env  

## Data Acquisition Step



1. Run the process_all_data.sh shell script to add the data into the mongodb

## Data Analysis Step

Each folder contains the model codes with different variations. 

In the LinearLogisticRegression folder, 