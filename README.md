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

### Dataset description
<br>

| Column Name | Description                                                                                                                           |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------|
| instant     | record index                                                                                                                          |
| dteday      | date                                                                                                                                  |
| season      | season (1:spring, 2:summer, 3:fall, 4:winter)                                                                                        |
| yr          | year (0: 2011, 1:2012)                                                                                                               |
| mnth        | month (1 to 12)                                                                                                                       |
| holiday     | weather day is holiday or not (extracted from [holiday schedule](http://dchr.dc.gov/page/holiday-schedule))                          |
| weekday     | day of the week                                                                                                                       |
| workingday  | if day is neither weekend nor holiday is 1, otherwise is 0                                                                           |
| schoolday   | if day is a normal school day is 1, otherwise is 0                                                                                   |
| weathersit  | 1: Clear, Few clouds, Partly cloudy, Partly cloudy<br>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist<br>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds<br>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog |
| temp        | Normalized temperature in Celsius. The values are divided to 41 (max)                                                                 |
| atemp       | Normalized feeling temperature in Celsius. The values are divided to 50 (max)                                                        |
| hum         | Normalized humidity. The values are divided to 100 (max)                                                                              |
| windspeed   | Normalized wind speed. The values are divided to 67 (max)                                                                             |
| casual      | count of casual users                                                                                                                 |
| registered  | count of registered users                                                                                                             |
| cnt         | count of total rental bikes including both casual and registered          

### Imports
This project has following libraries:
```python

# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce
import re
import math
import calendar
import graphviz
import warnings
from tabulate import tabulate
import time
import optuna
import pickle

# Machine Learning Libraries
from sklearn import tree
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error, max_error, make_scorer
from yellowbrick.model_selection import RFECV, LearningCurve
from yellowbrick.regressor import PredictionError, ResidualsPlot
import xgboost as xgb

# Set random state and cross-validation folds
random_state = 2024
n_splits = 10
cv = 10

# Warnings handling
warnings.filterwarnings("ignore")

# Set seaborn style
sns.set_style("whitegrid")
```
### Versions   
```
pandas version: 2.1.4
numpy version: 1.23.5
matplotlib version: 3.8.0
seaborn version: 0.13.2
scikit-learn version: 1.3.2
```   

### Results

#### Train and test results:

| Metric             |   Train Score |   Test Score |
|--------------------|---------------|--------------|
| Execution Time (s) |        0.0055 |       0.0055 |
| MAE                |      528.9499 |     562.3931 |
| RMSE               |      739.7028 |     756.5922 |
| R^2                |        0.8523 |       0.8502 |
| Adjusted R^2       |        0.8467 |       0.8248 |
| MAPE               |       44.9667 |      18.8200 |
| Max Error          |     3596.1236 |    3646.1657 |
<br>

**Execution Time (s)**: Both the training and testing times are very low (0.0055 seconds), indicating that the model trains and predicts quickly.

**MAE (Mean Absolute Error)**: The MAE measures the average absolute difference between the predicted and actual values. A lower MAE indicates better performance. The MAE on the test set (562.3914) is slightly higher than on the training set (528.9503), but the difference is not substantial.

**RMSE (Root Mean Squared Error)**: RMSE measures the average squared difference between the predicted and actual values, taking the square root to bring the metric back to the original scale. Again, lower values are better. Like MAE, the RMSE on the test set (756.5922) is slightly higher than on the training set (739.7028).

**R^2 (Coefficient of Determination)**: R-squared represents the proportion of variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, with higher values indicating better fit. Both the training (0.8523) and testing (0.8502) R-squared values indicate that the model explains a good amount of the variance in the data.

**Adjusted R^2**: 
Adjusted R-squared is similar to R-squared, but it adjusts for the number of predictors in the model. It penalizes the addition of unnecessary predictors that do not improve the model significantly. The testing adjusted R-squared (0.8248) is lower than the training adjusted R-squared (0.8467), indicating a potential overfitting issue or that some predictors in the model may not be contributing meaningfully to the prediction.

**MAPE (Mean Absolute Percentage Error)**: 
MAPE measures the percentage difference between the predicted and actual values. Lower values are better. It's important to note that MAPE is sensitive to outliers. In this case, the MAPE on the test set (18.8199%) is substantially lower than on the training set (44.9667%), which could indicate that the model performs better on the test data in terms of relative error.   

**Max Error**:
Max Error simply represents the maximum difference between predicted and actual values. Lower values are desirable. Both the training and testing max errors are relatively high, indicating that there are instances where the model performs poorly.
<br>

### Feature Importance:

|   Feature      |   Coefficient  |   AbsCoefficient  |
|----------------|----------------|-------------------|
| season_Winter  |   -2028.166009 |   2028.166009     |
| month_Nov      |   -1800.578534 |   1800.578534     |
| season_Spring  |   -1102.392630 |   1102.392630     |
| month_Dec      |   -819.433863  |   819.433863      |
| day            |    794.531195  |   794.531195      |
| year_2011.0    |   -669.571452  |   669.571452      |
| month_Oct      |    610.614878  |   610.614878      |
| month_Apr      |   -567.694729  |   567.694729      |
| month_May      |   -505.641698  |   505.641698      |
| month_Jul      |   -495.989634  |   495.989634      |
| year_2012.0    |   -475.406866  |   475.406866      |
| season_Fall    |    424.937125  |   424.937125      |
| month_Mar      |    413.235976  |   413.235976      |
| holiday_yes    |    337.438154  |   337.438154      |
| month_Jun      |    312.715197  |   312.715197      |
| holiday_no     |   -288.708558  |   288.708558      |
| month_Aug      |   -250.104412  |   250.104412      |
| month_Jan      |   -213.334701  |   213.334701      |
| month_Feb      |    194.243545  |   194.243545      |
| month_Sep      |   -107.207044  |   107.207044      |
| season_Summer  |    -72.135847  |    72.135847      |
<br>

*Top 5 Features*:
- Year 2011
- Weather Condition: Light snow & Rain
- Season: Spring
- Temperature Category: Cold
- Temperature Category: Hot

*Significant Features*:
Various months, weekdays, wind speed, humidity, and holiday categories also show notable impacts on the target variable.
<br>
