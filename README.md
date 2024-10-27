**Abaout Dataset;**

**California Housing**
This is a dataset obtained from the StatLib repository. Here is the included description:
S&P Letters Data
We collected information on the variables using all the block groups in California from the 1990 Cens us. In this sample a block group on average includes 1425.5 individuals living in a geographically co mpact area. Naturally, the geographical area included varies inversely with the population density. W e computed distances among the centroids of each block group as measured in latitude and longitude. W e excluded all the block groups reporting zero entries for the independent and dependent variables. T he final data contained 20,640 observations on 9 variables. The dependent variable is ln(median house value).
                            Bols    tols

INTERCEPT		       11.4939 275.7518

MEDIAN INCOME	       0.4790  45.7768

MEDIAN INCOME2	       -0.0166 -9.4841

MEDIAN INCOME3	       -0.0002 -1.9157

ln(MEDIAN AGE)	       0.1570  33.6123

ln(TOTAL ROOMS/ POPULATION)    -0.8582 -56.1280

ln(BEDROOMS/ POPULATION)       0.8043  38.0685

ln(POPULATION/ HOUSEHOLDS)     -0.4077 -20.8762

ln(HOUSEHOLDS)	       0.0477  13.0792

The file contains all the the variables. Specifically, it contains median house value, med ian income, housing median age, total rooms, total bedrooms, population, households, latitude, and lo ngitude in that order.

**Reference**

Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions, Statistics and Probability Letters, 33 (1997) 291-297.

**![BOSTON HOUSİNG](https://github.com/user-attachments/assets/a61b1c7e-6da8-47e5-b33d-5fec952c4140)**

**GOAL;**
Goal: The objective is to predict house prices using various features, such as the number of rooms, the size of the house, location, proximity to key amenities, and neighborhood characteristics. By analyzing these factors, the aim is to create a predictive model that estimates house prices accurately. This helps in understanding how different variables impact property values and provides insights for potential buyers, sellers, or real estate professionals to make informed decisions. The ultimate goal is to develop a reliable machine learning model that can generalize well to unseen data and offer accurate predictions based on the dataset.

**ALGORİTHM**


**Linear Regression** is a statistical method used for modeling the relationship between a dependent variable (target) and one or more independent variables (features). It assumes a linear relationship, meaning that changes in the independent variables result in proportional changes in the dependent variable.

**![lineer regresyon hausing price](https://github.com/user-attachments/assets/af719ccd-a790-4dbc-bba8-68338bc904a0)**

Linear Regression Model Performance:

Mean Squared Error (MSE): 0.56

R-squared Score: 0.58

Mean Absolute Error (MAE): 0.53

**Random Forest** is an ensemble learning method primarily used for classification and regression tasks. It combines the predictions from multiple decision trees to improve the overall performance and robustness of the model

**![hause price Random Forest](https://github.com/user-attachments/assets/d40401a7-b64c-48b7-aa69-dcc2669947ec)**

Random Forest Model Performance:

Mean Squared Error (MSE): 0.26

R-squared Score: 0.81

Mean Absolute Error (MAE): 0.33


**Gradient Boosting Regressor** is an ensemble learning technique that builds a predictive model in a stage-wise fashion from a collection of weak learners, typically decision trees. It is particularly effective for regression tasks, where it aims to predict a continuous target variable
![hause price gradient boosting resressor](https://github.com/user-attachments/assets/f0d6ec20-68bf-47c1-bfef-2e87c331fdf8)

Gradient Boosting Regressor Model Performance:

Mean Squared Error (MSE): 0.23

R-squared Score: 0.82

Mean Absolute Error (MAE): 0.32
