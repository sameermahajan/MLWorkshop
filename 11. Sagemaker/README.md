
# Sagemaker-pipeline-1 (basic pipeline)

- environment pipeline
- data preparation
- modelling
- deployement
- evaluation
- prediction


## Environment Setup
1. get aws role
2. region and other aws configuational artifacts

## Data Preparation
upload the raw data to S3 bucket<br />
sagemaker conatiner based training read data only from S3<br />

## Modelling
train sagemaker based xgboost model

## Deployement
Deploy the xgboost model as an endpoint
Mention:<br />
- instance_type : 'ml.m4.xlarge'
- initial_instance_count : 1
After this step we have successfully deployed ML model as an Endpoint

## Evaluation
1. generate confusion martrix<br />
2. Pr curve<br />

## Prediction
Predict using endpoint







