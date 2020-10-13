# Inference Pipeline using Scikit-learn model

![](pipeline-Sagemaker.png)


## step 1 : build sklearn pipeline
Pipeline setup<br />
### 1. sklearn data pre-processor model
 * numeric : imputation, scaling
 * categoric : imputation, one-hot-encoding
 * also perform batch transformation of train/test data to be used for ml_model training
 
#### local training
Idea behind training sklearn model locally is to verify the model training script **pre_processing_script.py** and make sure that there are not any anomalies present in the script.<br />
*It will require raw data in local dir*<br />
This approach gives following benefits:
1. Time saving. Container based training take considerable amount of time.
2. Cost saving. Container based training is billable as per the selected instance.

#### container based training
- once local run is succesfull, train the sagemaker-container based model
- pre_processing_script.py is wokring fine
- train sklearn_preprocessor container

**setup**<br />
configure the training job<br />
Mention:<br />
1. entry_point : script name
2. train_instance_type : example "ml.c4.xlarge"

**training**<br />
*It will require raw data in S3 bucket*
launch container based model training
<br /><br />



### 2. sklearn classification Model
train sklearn RandomforestClassifier model
#### local training
*It will require pre-processed data in local dir.*<br />
Training sklearn model locally, to verify the model training script **model_script.py** and make sure that there are not any anomalies present in the script.<br />


#### container based training
**setup**<br />
configure the training job<br />
Mention:<br />
1. entry_point : script name
2. train_instance_type : example "ml.c4.xlarge"

**training**<br />
*It will require raw data in S3 bucket*<br />
launch container based model training
<br /><br />


<br /><br />
## step 2 : Serial Inference Pipeline
* raw_data(S3) --> [preprocessing ==> ml_model] --> prediction
### setup
setup the PipelineModel:
Below trained container models are to be used in pipeline:
- pre-process transformer model<br />
- ml classification model<br />

### deploy endpoint
Mention:<br />
1. instance_type : 'ml.m4.xlarge'<br />
2. initial_instance_count : 1<br />
After this step we have successfully deployed ML inference pipeline as an Endpoint<br />


## step 3 : prediction using point
send raw data to the endpoint<br />
and get the prediction<br />


## step 4 : model insights
### retrieve model training artifacts
- pipeline model
- confusion matrix
- pr curve

### SHAP analysis
- Force plots for individual data points
- Summary statistics for all data points
- Feature importance