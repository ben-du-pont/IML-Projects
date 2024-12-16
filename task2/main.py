import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("task2/train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("task2/test.csv")

    print("Test data:")
    print("Shape:", test_df.shape)
    print(test_df.head(2))
    print('\n')

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # Define indices for feature selection
    feature_indices = [1] + list(range(3, 11, 1))  # Skip 'season' and 'price_CHF'

    # Extract raw arrays
    x_train_season = train_df.values[:, 0]  # Season column
    x_train = train_df.values[:, feature_indices]  # Input features excluding 'season'
    y_train = train_df.values[:, 2].astype(float)  # Target 'price_CHF'
    x_test_season = test_df.values[:, 0]  # Season column
    x_test = test_df.values[:, 1:]  # Input features excluding 'season'

    # Combine for imputation
    iter_imp = IterativeImputer(missing_values=np.nan, imputation_order='ascending')
    iter_imp.fit(np.concatenate((x_train, x_test), axis=0))
    x_train_imputed = iter_imp.transform(x_train)
    x_test_imputed = iter_imp.transform(x_test)

    # Filter training data to exclude rows with NaN in y_train
    valid_rows = ~np.isnan(y_train)
    y_train_imputed = y_train[valid_rows]
    x_train_imputed = x_train_imputed[valid_rows, :]
    x_train_season = x_train_season[valid_rows]

    # One-hot encode 'season'
    encoder = OneHotEncoder(sparse_output=False)
    x_train_season_onehot = encoder.fit_transform(x_train_season.reshape(-1, 1))
    x_test_season_onehot = encoder.transform(x_test_season.reshape(-1, 1))

    # Concatenate one-hot encoded 'season' with imputed data
    X_train = np.concatenate((x_train_imputed, x_train_season_onehot), axis=1)
    X_test = np.concatenate((x_test_imputed, x_test_season_onehot), axis=1)
    y_train = y_train_imputed

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    
    kernel = RationalQuadratic()
    clf = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)

    # Predict on the test set
    y_pred = clf.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


if __name__ == "__main__":

    # Data loading
    print("\n")
    print("Loading and preprocessing data...")
    X_train, y_train, X_test = data_loading()

    # The function retrieving optimal LR parameters
    print("\n")
    print("Fitting the model and predicting...")
    y_pred=modeling_and_prediction(X_train, y_train, X_test)

    # Save results in the required format
    print("\n")
    print("Saving results...")

    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('task2/results.csv', index=False)

    print("\nResults file successfully generated!")

