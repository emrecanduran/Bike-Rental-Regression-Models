### Bike-Rental-Regression-Models

## Bike_Rentals.xlsx

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

The Random Forest model employed in this project utilizes random split hyperparameters, which create an ensemble of decision trees trained on diverse subsets of the training data. 

Evaluation of the model's performance is done using key metrics such as AUC-ROC, a widely-used measure that assesses the classifier's ability to distinguish between classes. The confusion matrix provides a comprehensive breakdown of the model's predictions, enabling a detailed analysis of true positives, true negatives, false positives, and false negatives. Classification reports offer a concise summary of precision, recall, F1 score, and support for each class. 

Additionally, the model provides insights into feature importance, facilitating feature selection, and understanding of the factors driving predictions. For further analysis, a single decision tree can be visualized to comprehend the underlying decision-making process.
