# Introduction to machine learning - Four projects

This repository consists of four projects that were part of the Introduction to Machine Learning lecture given by Prof. Dr. Andreas Krause and Prof. Dr. Fan Yang at ETH Zurich in spring semester 2023.
These projects counted for 30% of the final grade of the course, and the remaining 70% was a final written exam. Full details of the course can be found on the official [webpage](https://las.inf.ethz.ch/teaching/introml-s23).

The projects were done in groups of up to 3 students, and the entire code in this repository was written by [Benjamin Dupont](https://github.com/ben-du-pont) and [Mateo Hamel](https://github.com/hamelmateo). The code was written in Python and used libraries such as numpy, pandas, scikit-learn, and pytorch, sometimes using a base template provided by the course staff.

## Summary of Projects

1. [Project 1: Linear Regression](#project-1-linear-regression)
    - Implemented cross-validation for ridge regression and explored feature transformations to improve model performance.

2. [Project 2: Prediction of electricity price in Switzerland](#project-2-prediction-of-electricity-price-in-switzerland)
    - Developed a model to predict electricity prices in Switzerland using data from other countries at various seasons of the year. Implemented an imputer to fill in the missing data and a gaussian process regressor to predict the price on a test set.

3. [Project 3: Classification of food preferences](#project-3-classification-of-food-preferences)
    - Created a classification model to determine food preferences based on image triplets. The project involved extracting embeddings from a pre-trained model and training a custom classifier neural network to determine for each image triplet (A, B, C) if A was closer to B or to C.

4. [Project 4: Transfer learning](#project-4-transfer-learning)
    - Applied transfer learning techniques to improve model accuracy on a new dataset.


### Project 1: Linear Regression
This first project was an introduction to linear regression and ridge regression and was subdivided into two tasks:

#### 1a - Cross validation for ridge regression

Implemented K-fold cross-validation with 10 folds to evaluate the performance of ridge regression for 5 different values of lambda. The RMSE was calculated for each lambda and averaged over the folds to determine the optimal regularization parameter. The process involved splitting the data into training and validation sets, fitting the model on the training set, and evaluating it on the validation set. The results were used to select the best lambda value that minimized the average RMSE.

#### Training Data Format

The training data is provided in a CSV format with the following columns:

```
y, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13
```

- `y`: Target variable
- `x1, x2, ..., x13`: Feature variables

Exact implementation and datasets can be found in the [`task1a`](./task1a) folder.



#### 1b - Ridge regression with transformed features
Need to transform the original features into linear, polynomial and exponential features and then use ridge regression to predict the target variable. The goal was to find the optimal weights of the regression model on these transofrmed features.
The original features were transformed into polynomial and exponential features to capture non-linear relationships in the data. Specifically, polynomial features up to degree 3 and exponential features were generated. The ridge regression model was then trained on these transformed features to predict the target variable `y`.

The implementation involved the following steps:

1. **Feature Transformation**: The original features were transformed into polynomial features and exponential features.
2. **Model Training**: Ridge regression was applied to the transformed features. The model was trained using the training data provided.
3. **Cross-Validation**: Cross-validation was performed to select the optimal regularization parameter lambda. The RMSE was calculated for each lambda and averaged over the folds to determine the optimal regularization parameter and therefore the best model.

The code for this task can be found in the [`task1b`](./task1b) folder. The training data is provided in a CSV format with the following columns:

```
Id, y, x1, x2, x3, x4, x5
```

- `Id`: Identifier for each data point
- `y`: Target variable
- `x1, x2, ..., x5`: Feature variables



### Project 2: Prediction of electricity price in Switzerland
This project involved predicting the electricity price in Switzerland using data from other countries. The dataset included electricity prices from various countries across different seasons, with some missing values. The implementation steps were as follows:

1. **Data Imputation**: Missing values in the dataset were imputed using statistical methods to ensure a complete dataset for model training.
2. **Model Training**: A Gaussian Process Regressor with a RationalQuadratic kernel was trained on the imputed dataset to predict the electricity prices in Switzerland.
3. **Evaluation**: The model was evaluated on the test set, and the predictions were saved in a results file.

The code for this project can be found in the [`task2`](./task2) folder. The training data is provided in a CSV format with the following columns:

| Column       | Description                        |
|--------------|------------------------------------|
| `season`     | Season of the year                 |
| `price_AUS`  | Electricity price in Australia     |
| `price_CHF`  | Electricity price in Switzerland (to be predicted in the test set)  |
| `price_CZE`  | Electricity price in Czech Republic|
| `price_GER`  | Electricity price in Germany       |
| `price_ESP`  | Electricity price in Spain         |
| `price_FRA`  | Electricity price in France        |
| `price_UK`   | Electricity price in the United Kingdom |
| `price_ITA`  | Electricity price in Italy         |
| `price_POL`  | Electricity price in Poland        |
| `price_SVK`  | Electricity price in Slovakia      |



### Project 3: Classification of food preferences

This project aimed to classify food preferences based on image triplets. The task was to determine which of two images (B or C) is more similar to a reference image (A). The dataset consisted of images of food items, and the training data was provided in the form of triplets, where it was always true that image A was closer to image B than to image C.

#### Data Format

The data was organized in the following structure:
- **Dataset Folder**: `/food`
- **Training Triplets**: Provided in a CSV file with columns `A`, `B`, and `C`, representing the file paths of the images in each triplet.

#### Goal

The goal was to train a model that could predict, for each triplet in the test set, whether image A is closer to image B than to image C. The predictions were binary, with `1` indicating that A is closer to B, and `0` indicating that A is closer to C.

#### Implementation

The implementation involved the following steps:

1. **Data Preprocessing**: 
    - Loaded the images from the dataset folder.
    - Resized and normalized the images to ensure consistency in input dimensions.

2. **Feature Extraction**:
    - Used a pre-trained convolutional neural network (CNN) to extract embeddings from the images. The embeddings captured the high-level features of the images.

3. **Model Training**:
    - Constructed a custom neural network classifier that took the embeddings of the triplets as input.
    - The network was trained to minimize a loss function that encouraged the model to correctly predict the similarity relationship in the training triplets.

4. **Evaluation**:
    - Evaluated the model on the test set by predicting the similarity relationship for each triplet.
    - Annotated the test triplets with `1` (true) or `0` (false) based on the model's predictions.

The code for this project can be found in the [`task3`](./task3) folder. The main implementation details are in the `main.py` file, which includes the data loading, model definition, training loop, and evaluation logic.

The training data is provided in a CSV format with the following columns:

| Column | Description                  |
|--------|------------------------------|
| `A`    | Number/File path of the reference image |
| `B`    | Number/File path of the first comparison image |
| `C`    | Number/File path of the second comparison image |

The training data label is assumed to be `1` for all triplets, indicating that image A is closer to image B than to image C.

The test data follows the same format, and the goal is to predict whether image A is closer to image B than to image C for each triplet.

### Project 4: 

Transfer learning 
