### Bike-Rental-Regression-Models

### bike_day_v2.xlsx

<p>The customer wishes to build a model to predict everyday at 15h00 the total number of bikes they will rent the following day. This will allow them not only to better allocate staff resources, but also to define their daily marketing budget in social media which is their principal form of advertisement.</p> 

## Dataset 

The dataset used for this project contains the following variables:

Bike_Rentals.xlsx dataset (732) 

- `instant`: record index
- `dteday` : date
- `season` : season (1:spring, 2:summer, 3:fall, 4:winter)
- `yr` : year (0: 2011, 1:2012)
- `mnth` : month (1 to 12)
- `holiday` : weather day is holiday or not (extracted from http://dchr.dc.gov/page/holiday-schedule)
- `weekday` : day of the week
- `workingday` : if day is neither weekend nor holiday is 1, otherwise is 0
- `schoolday` : if day is a normal school day is 1, otherwise is 0
- `weathersit` :
	- 1: Clear, Few clouds, Partly cloudy, Partly cloudy
	- 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
	- 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
	- 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
- `temp`: Normalized temperature in Celsius. The values are divided to 41 (max)
- `atemp`: Normalized feeling temperature in Celsius. The values are divided to 50 (max)
- `hum`: Normalized humidity. The values are divided to 100 (max)
- `windspeed`: Normalized wind speed. The values are divided to 67 (max)
- `casual**: count of casual users
- `registered`: count of registered users
- `cnt`: count of total rental bikes including both casual and registered

## Model Building

The Random Forest, XGBoost, GradientBoosting and Lasso Regression models are employed in this project utilizes Optuna hyperparameters. 

The evaluation of the model's performance entails a comprehensive analysis, using metrics such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared (R^2), Adjusted R-squared, Mean Absolute Percentage Error (MAPE), and Max Error, alongside the consideration of execution time. This evaluation process is further enriched through the examination of learning curves, residuals, and prediction error plots, along with the assessment of train and test scores. 

Additionally, the model provides insights into feature importance, facilitating feature selection, and understanding of the factors driving predictions. 
