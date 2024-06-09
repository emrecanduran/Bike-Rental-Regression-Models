<div align="center">

# Daily Total Bikes Prediction Project

</div>

## CRISP-DM Methodology

CRISP-DM is like our roadmap for exploring and understanding data. It breaks down complex data projects into manageable steps, starting from understanding what the business needs are, all the way to deploying our solutions. It's super helpful because it keeps us organized and ensures we're on track with our goals. Plus, it's not just for one specific industry, so we can apply it to all sorts of projects, which is awesome for learning and building our skills. It's basically our guide to navigating the world of data mining!

<p align="center">
  <img width="550" height="550" src="images/CRISP-DM.png" alt="CRISP-DM process diagram">
</p>

<p align="center">
  CRISP-DM process diagram
</p>

*    **Business Understanding**: determine business objectives; assess situation; determine data mining goals; produce project plan
*    **Data Understanding**: collect initial data; describe data; explore data; verify data quality
*    **Data Preparation** (generally, the most time-consuming phase): select data; clean data; construct data; integrate data; format data
*    **Modeling**: select modeling technique; generate test design; build model; assess model
*    **Evaluation**: evaluate results; review process; determine next steps
*    **Deployment**: plan deployment; plan monitoring and maintenance; produce final report; review project (deployment was not required for this project)

[Reference](https://github.com/mbenetti/CRISP-DM-Rossmann/blob/master/README.md)

### **Overview**
<p>The customer whishes to build a model to predict everyday at 15h00 the total number of bikes they will rent the following day. This will allow them not only to better allocate staff resources, but also to define their daily marketing budget in social media which is their principal form of advertisement.</p>

### Model building

<p>To achieve the objective, it is followed a systematic approach, CRISP-DM, that involves several stages. It is started by preparing the data, cleaning, and organizing it for analysis. Next, perform exploratory data analysis (EDA) to gain insights into the dataset and identify any patterns or trends. Once I have a thorough understanding of the data, it is proceed to train and evaluate predictive models using 4 different machine learning techniques with their best parameters such as: <p>

- Random Forest Regressor
- XGBoost 
- GradientBoosting 
- Lasso Regression

I tried to explore various models from different families, including bagging techniques like RandomForestRegressor, boosting methods such as XGBoost and GradientBoosting, as well as Lasso Regression.
