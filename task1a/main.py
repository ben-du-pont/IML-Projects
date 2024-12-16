import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

# The goal of this subtask is to implement a 10-fold cross-validation for ridge regression.
# This is done by loading the data, and then fitting the ridge regression on the training set for every lambda, and doing this for every fold created by KFold.
# This gives us an average RMSE for our model for every lambda, averaged over all folds.
# The results (i.e the RMSE we calculated for the 5 different lambdas) are saved in the required format in the results.csv file.

def fit(X, y, lam):
    """
    This function receives training data points, then fits the ridge regression on this data
    with regularization hyperparameter lambda. The weights w of the fitted ridge regression
    are returned. 

    Parameters
    ----------
    X: matrix of floats, dim = (135,13), inputs with 13 features
    y: array of floats, dim = (135,), input labels)
    lam: float. lambda parameter, used in regularization term

    Returns
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression
    """

    w = np.zeros((13,))
    w = Ridge(alpha = lam,fit_intercept = False).fit(X,y).coef_

    assert w.shape == (13,)
    return w


def calculate_RMSE(w, X, y):
    """This function takes test data points (X and y), and computes the empirical RMSE of 
    predicting y from X using a linear model with weights w. 

    Parameters
    ----------
    w: array of floats: dim = (13,), optimal parameters of ridge regression 
    X: matrix of floats, dim = (15,13), inputs with 13 features
    y: array of floats, dim = (15,), input labels

    Returns
    ----------
    RMSE: float: dim = 1, RMSE value
    """

    

    RMSE = 0

    y_predict = X @ w
    RMSE = root_mean_squared_error(y, y_predict)

    assert np.isscalar(RMSE)
    return RMSE


def average_LR_RMSE(X, y, lambdas, n_folds):
    """
    Main cross-validation loop, implementing 10-fold CV. In every iteration (for every train-test split), the RMSE for every lambda is calculated, 
    and then averaged over iterations.
    
    Parameters
    ---------- 
    X: matrix of floats, dim = (150, 13), inputs with 13 features
    y: array of floats, dim = (150, ), input labels
    lambdas: list of floats, len = 5, values of lambda for which ridge regression is fitted and RMSE estimated
    n_folds: int, number of folds (pieces in which we split the dataset), parameter K in KFold CV
    
    Returns
    ----------
    avg_RMSE: array of floats: dim = (5,), average RMSE value for every lambda
    """

    RMSE_mat = np.zeros((n_folds, len(lambdas)))

    folds = KFold(n_folds)

    for i, (train_idx, test_idx) in enumerate(folds.split(X,y)):

        for j, lam in enumerate(lambdas):
            w = fit(X[train_idx],y[train_idx],lam)
            RMSE_mat[i,j] = calculate_RMSE(w, X[test_idx], y[test_idx])
        

    avg_RMSE = np.mean(RMSE_mat, axis=0)

    assert avg_RMSE.shape == (5,)
    return avg_RMSE



if __name__ == "__main__":

    # Data loading
    print("\n")
    print("Loading data...")
    data = pd.read_csv("task1a/train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns="y")

    # Print sample of data
    print("\n")
    print("Data sample:")
    print(data.head())

    X = data.to_numpy()

    # The function calculating the average RMSE
    lambdas = [0.1, 1, 10, 100, 200]
    n_folds = 10
    print("\n")
    print("Calculating RMSE...")
    avg_RMSE = average_LR_RMSE(X, y, lambdas, n_folds)

    print("\n")
    print("Average RMSE for every lambda in", lambdas, ":")
    print(avg_RMSE)

    # Save results in the required format
    print("\n")
    print("Saving results...")
    np.savetxt("./task1a/results.csv", avg_RMSE, fmt="%.12f")
