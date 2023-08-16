# Dynamic Risk Assessment System
- Project: **Dynamic Risk Assessment System** of ML DevOps Engineer Nanodegree Udacity.

## Project Overview
This project builds an Attrition Detection ML System that is regularliy scored and monitored for Model Drift.
 
## Project Description
ML models monitoring is a vital part of ML project lifecycle and an important steps in MLOps. The importance of monitoring stems from the need to watch out for model drift.

This project aim to build processes to regularliy perform the following:
- Ingest new data.
- Check for model drift.
- Re-train.
- Re-deploy.
- Monitor with REST API.
- Report on the performance of the ML system.
- Automating the entire process with cron job.

### Model Drift
Model Drift refers to the phenomenon that a model tends to perform worse over time when tested on new data. Model drift can be attributed to several factors including ever-chainging industry dynamics, changes in the distribution of dependant and independant variables, and chainging in the relationships between dependant and independant variables.

A simple way of detecting model drift is performing **Raw Comparision Test** which is simply comparing the current model score with the last recorded score given by the metric of choice. Model drift is said to have occured when the new score is worse. This project uses this simple technique to check for model drift.

An Alternative and less sensitive method is **Parametric Significance Test** which check if the new score is worse than 2-standard-deviations away from the mean of the previous scores.

If the data does not assume a normal distribution, then **Non-Parametric Significance Test** can be used, which check whether the current score is an outlier, that is, the new score is 1.5*(Interquartile Range) above the 75th percentile or 1.5*(Interquartile Range) below the 25th percentile.

### Data Integrity and Stability
An important aspect of monitoring is to check the **integrity** and **stability** of new data.
**Data Integrity** issues arise when data have a high percentage of missing values, missing values of critical feature, or data is invalid, e.g. certain features is outside the allowed range.
On the other hand, **Stability** refers to the fact that the new data is statistically similar to the data initially used for training.
Data Integrity issues can be resolved with various imputation techniques, while Data Stability issues can be resolved by checking the data source, or retraining the model.

### Execution Time Monitoring
Another aspect of monitoring is to calculate and record the time required for various steps of the pipeline, such as data ingestion and model training.  

### Dependencies Monitoring
To ensure smooth operation of the pipeline, dependencies are monitored by keeping record of each package version used in the project and the latest version available for that package.

### REST API
The above-mentioned monitoring processes are made available via a REST API using Flask, see `app.py`. The API is provided to offer a simple interface to various **stakeholders** who are interested in performing one or several operations including inference, scoring, summary statistics and diagnostics.

