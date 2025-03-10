### README

## Directory Structure

The project comprises 2 steps - Data Acquisition and Data Analysis. 
The directory is structured accordingly. The DataAcquisition folder contains different data sources and utility codes that preprocesses them and stores them to the mongodb database.

The **Yelp** folder contains the code that uses Yelp API to query the store data in the Manhattan area. All of the data gets stored into the mongodb database.
In order to verify the data is correctly retrieved, there are different visualization code that read from the mongodb and visualizes them on the NYC map.

./DataAcquisition/Yelp/filter_by_data.py number of lines : 52

This code can be used to access shop data from the database and filter out invalid data.

./DataAcquisition/Yelp/defining_success.py number of lines : 68

This code can be used to compute the success score that I defined and update the database.

./DataAcquisition/Yelp/visualize_all_data.py number of lines : 171

This code can be used to visualize all the data in the database.

./DataAcquisition/Yelp/config.py number of lines : 11

This code has some utility code for the database set up.

./DataAcquisition/Yelp/get_all_categories.py number of lines : 31

This code has a utility code for reading all the category information from the database.

./DataAcquisition/Yelp/visualize_store_data.py number of lines : 77

This code can be used to visualize the store data.

./DataAcquisition/Yelp/yelp_automated_queries.py number of lines : 138

This code can be used to retrieve the store data using the yelp api. 

./DataAcquisition/Yelp/yelp_data_playground.py number of lines : 119

This code can be used to play with the stored yelp data from the database.

./DataAcquisition/Yelp/parse_categories.py number of lines : 25

This code can be used to parse the categories information from a text file to a csv file. This csv file is used in making the queries using the Yelp api.

./DataAcquisition/Yelp/filter_by_location.py number of lines : 181

This code can be used to filter the data by location.

./DataAcquisition/Yelp/categories_to_json.py number of lines : 21

This code can be used to generate the category json file.

./DataAcquisition/Yelp/visualize_manhattan_data.py number of lines : 168

This code can be used to visualize the manhattan data on the map.

The **Complaints** folder contains the complaints reported at different locations. The process code counts the number of complaints near each shop and stores the data to the database.

./DataAcquisition/Complaints/process_data.py number of lines : 171

This code processes the complaints data and stores them to the database.

The **Parks** folder contains the park data in the new york city. The code finds the k closest parks near each shop and stores them into the database. There is a code that visualizes the relationship for debugging.

./DataAcquisition/Parks/process_data.py number of lines : 108

This code processes the Parks data and stores them to the database.

./DataAcquisition/Parks/visualize_close_parks.py number of lines : 83

This code visualizes the close parks to the shop data.

The **PolicePrecincts** folder contains the police precinct location data. Similarly, the code finds the k closest police stations near each shop and updates the database.

./DataAcquisition/PolicePrecints/geo_util.py number of lines : 78

This code has the geo utils code to find the closest location.

./DataAcquisition/PolicePrecints/visualize_closest_precints.py number of lines : 93

This code visualizes the police precincts data associated to the stores.

./DataAcquisition/PolicePrecints/process_data.py number of lines : 84

This processes the data and updates the database.

The **RatSighting** has the data with ratsighting reports. First, the rat sighting data is aggregated asa count for each zip code and then associated with the shop data. 

./DataAcquisition/RatSighting/aggregate_sighting_count.py number of lines : 120

This aggregates the rat sighting data per each zip code.

./DataAcquisition/RatSighting/process_data.py number of lines : 145

This processes the aggregated count data and updates the database.

The **Schools** folder contains the school location data. The same opration to find the closest school is performed and stored in the database.

./DataAcquisition/Schools/process_data.py number of lines : 158

This code processes the school data and updates the database.

The **Subway** folder contains the subway data. Similar operation is performed.

./DataAcquisition/Subway/process_data.py number of lines : 122

THis processes the subway data and updates the database.

./DataAcquisition/Subway/visualize_stations.py number of lines : 94

This visualizes the subway stations.

The **WalkingTraffic** contains the walking traffic data. I processed and visualized the data but the data was confined to the Time Square location. This is not used in the analysis.

./DataAcquisition/WalkingTraffic/process_data.py number of lines : 134

This processes the data.

./DataAcquisition/WalkingTraffic/walking_traffic_processing.py number of lines : 139

This visualizes the data.

In the **DataAnalysis** folder, all the modeling codes are stored. 

I used different types of modeling, each of which is stored in a separate folder.

The **DecisionTree** folder contains the decision tree modeling code with the resulting tree visualized.

./DataAnalysis/DecisionTree/decision_tree_with_all_features_with_categories.py number of lines : 91

This code models the manhattan data using the decision tree model. 

./DataAnalysis/DecisionTree/data_reader.py number of lines : 23

Utility code to read the data from the db.

./DataAnalysis/DecisionTree/data_preprocessing.py number of lines : 73

Utility code to preprocesses the data read.

The **LinearLogisticRegression** folder contains the modeling code for the linear and logistic regressions. 

./DataAnalysis/LinearLogisticRegression/linear_logistic_regression_using_all_features_with_normalization.py number of lines : 67

Linear regression and logistic regression modeling after normalization is applied.

./DataAnalysis/LinearLogisticRegression/linear_logistic_regression_using_all_features_without_normalization.py number of lines : 114

Linear regression and logistic regression modeling without the normalization step.


./DataAnalysis/LinearLogisticRegression/linear_regression_util.py number of lines : 43

Utility code for the regression model training and evaluation.

./DataAnalysis/LinearLogisticRegression/data_reader.py number of lines : 23

Utility code to read the data from the db.

./DataAnalysis/LinearLogisticRegression/linear_logistic_regression_using_all_features_with_categories_cv_collinearity_no_outlier.py number of lines : 114

Linear regression and logistic regression modeling with collinearity analysis and outlier rejection. This code failed to run.

./DataAnalysis/LinearLogisticRegression/linear_logistic_regression_using_all_features_with_categories_cv.py number of lines : 134

Linear regression and logistic regression modeling with the category data and cross validation.

./DataAnalysis/LinearLogisticRegression/linear_logistic_regression_using_all_features_with_categories.py number of lines : 140

Linear regression and logistic regression modeling with the category data.

./DataAnalysis/LinearLogisticRegression/linear_logistic_regression_using_all_features_with_categories_cv_collinearity.py number of lines : 158

Linear regression and logistic regression modeling with collinearity analysis. This code failed to run.

./DataAnalysis/LinearLogisticRegression/data_preprocessing.py number of lines : 73

Utility code to preprocesses the data read.

**NeuralNetwork**

This contains the neural network modeling works.

./DataAnalysis/NeuralNetwork/neural_network_wider.py number of lines : 119

This makes NN model that is wide. This code takes about 3~5 minutes per epoch with insignificant loss improvement. 

./DataAnalysis/NeuralNetwork/neural_network_resnet.py number of lines : 180

This makes the RESNET model. 

./DataAnalysis/NeuralNetwork/neural_network_deeper.py number of lines : 179

THis makes a deep neural network model. Training it was faster than the wider one but the loss reduction was negligible. 

./DataAnalysis/NeuralNetwork/data_reader.py number of lines : 23

Data reader utility.

./DataAnalysis/NeuralNetwork/neural_network.py number of lines : 160

Neural network modeling work.

./DataAnalysis/NeuralNetwork/data_preprocessing.py number of lines : 73

Data preprocessing utility.


**OldDrafts**

This contains some old codes I ran.

**SVM**

This contains the support vector machine modeling codes and results.

./DataAnalysis/SVM/svm_using_all_features_with_categories.py number of lines : 95

This makes the support vector machine model.

./DataAnalysis/SVM/data_reader.py number of lines : 23

Data reader utility.

./DataAnalysis/SVM/data_preprocessing.py number of lines : 73

Data preprocessing utility.

**Docker**
In order to simplify the operation, I made the docker container for all the operations. The docker files are stored in the Docker directory.


## Data File Description



## How to set up

1. Set up the docker.

docker build -t business_predictor_env .

2. Run the docker container

docker run --network="host" -v $(pwd):/projects -it business_predictor_env  

## Data Acquisition Step

1. Run the process_all_data.sh shell script to add the data into the mongodb

## Data Analysis Step

For each model, one can run the modeling code to read the data from the mongodb and train a model.