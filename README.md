# ETH IML 2023 course solution repository
This repository contains the code we used to solve the projects in the course given at ETH as "Introduction to Machine Learning" during Spring 2023.

### Project 1: Linear Regression
#### 1a - Cross validation for ridge regression
Code in task 1a folder.
train data of the form 
y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13

Need to use Kfolds on 5 different values of lambda to calculate the RMSE for each lambda and average it over 10 folds

#### 1b - Ridge regression with transformed features
Code in task 1b folder.
train data of the form
Id,y,x1,x2,x3,x4,x5

Need to transform the original features into linear, polynomial and exponential features and then use ridge regression to predict the target variable. The goal was to find the optimal weights of the regression model on these transofrmed features.

### Project 2: Prediction of electricity price in Switzerland
Code in task 2 folder.
train data of the form
season,price_AUS,price_CHF,price_CZE,price_GER,price_ESP,price_FRA,price_UK,price_ITA,price_POL,price_SVK
but sometimes some data is missing

Need to predict the price of electricity in Switzerland using the prices of electricity in other countries from test.py and output in results file 

### Project 3: Classification of food preferences
Code in task 3 folder.
data of the form of images:
in dataset / food
training data of the form of triplets of images and the goal is to predict which of the two images is more similar to the first one
In the train triplets it is always true that when evaluating image A it is closer to image B than it is to image C
In the test triplets, you need to annotate with 1 (true) or 0 (false) if image A is closer to image B than it is to image C

### Project 4: 
