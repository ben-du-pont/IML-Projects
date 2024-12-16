import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge


def transform_data(X):
    """
    This function transforms the 5 input features of matrix X (x_i denoting the i-th component of X) 
    into 21 new features phi(X) in the following manner:
    5 linear features: phi_1(X) = x_1, phi_2(X) = x_2, phi_3(X) = x_3, phi_4(X) = x_4, phi_5(X) = x_5
    5 quadratic features: phi_6(X) = x_1^2, phi_7(X) = x_2^2, phi_8(X) = x_3^2, phi_9(X) = x_4^2, phi_10(X) = x_5^2
    5 exponential features: phi_11(X) = exp(x_1), phi_12(X) = exp(x_2), phi_13(X) = exp(x_3), phi_14(X) = exp(x_4), phi_15(X) = exp(x_5)
    5 cosine features: phi_16(X) = cos(x_1), phi_17(X) = cos(x_2), phi_18(X) = cos(x_3), phi_19(X) = cos(x_4), phi_20(X) = cos(x_5)
    1 constant features: phi_21(X)=1

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features

    Returns
    ----------
    X_transformed: array of floats: dim = (700,21), transformed input with 21 features
    """
    X_transformed = np.zeros((700, 21))

    # Proceed with the transformation of the data as defined in the function definition
    for i in range(700):
        X_transformed[i,0] = X[i,0] # phi_1(X) = x_1
        X_transformed[i,1] = X[i,1] # phi_2(X) = x_2
        X_transformed[i,2] = X[i,2] # phi_3(X) = x_3
        X_transformed[i,3] = X[i,3] # phi_4(X) = x_4
        X_transformed[i,4] = X[i,4] # phi_5(X) = x_5
        X_transformed[i,5] = np.square(X[i,0]) # phi_6(X) = x_1^2
        X_transformed[i,6] = np.square(X[i,1]) # phi_7(X) = x_2^2
        X_transformed[i,7] = np.square(X[i,2]) # phi_8(X) = x_3^2
        X_transformed[i,8] = np.square(X[i,3]) # phi_9(X) = x_4^2
        X_transformed[i,9] = np.square(X[i,4]) # phi_10(X) = x_5^2
        X_transformed[i,10] = np.exp(X[i,0]) # phi_11(X) = exp(x_1)
        X_transformed[i,11] = np.exp(X[i,1]) # phi_12(X) = exp(x_2)
        X_transformed[i,12] = np.exp(X[i,2]) # phi_13(X) = exp(x_3)
        X_transformed[i,13] = np.exp(X[i,3]) # phi_14(X) = exp(x_4)
        X_transformed[i,14] = np.exp(X[i,4]) # phi_15(X) = exp(x_5)
        X_transformed[i,15] = np.cos(X[i,0]) # phi_16(X) = cos(x_1)
        X_transformed[i,16] = np.cos(X[i,1]) # phi_17(X) = cos(x_2)
        X_transformed[i,17] = np.cos(X[i,2]) # phi_18(X) = cos(x_3)
        X_transformed[i,18] = np.cos(X[i,3]) # phi_19(X) = cos(x_4)
        X_transformed[i,19] = np.cos(X[i,4]) # phi_20(X) = cos(x_5)
        X_transformed[i,20] = 1.0 # phi_21(X)=1

    assert X_transformed.shape == (700, 21)
    return X_transformed


def fit(X, y):
    """
    This function receives training data points, transform them, and then fits the linear regression on this 
    transformed data. Finally, it outputs the weights of the fitted linear regression. 

    Parameters
    ----------
    X: matrix of floats, dim = (700,5), inputs with 5 features
    y: array of floats, dim = (700,), input labels)

    Returns
    ----------
    w: array of floats: dim = (21,), optimal parameters of linear regression
    """
    w = np.zeros((21,))
    X_transformed = transform_data(X)

    w = Ridge(325, fit_intercept = False, positive=False).fit(X_transformed, y).coef_

    assert w.shape == (21,)
    return w


if __name__ == "__main__":

    # Data loading
    print("\n")
    print("Data loading...")
    data = pd.read_csv("task1b/train.csv")
    y = data["y"].to_numpy()
    data = data.drop(columns=["Id", "y"])

    # print data samples
    print("\n")
    print("Data samples:")
    print(data.head())

    X = data.to_numpy()

    # The function retrieving optimal LR parameters
    print("\n")
    print("Fitting the model...")
    w = fit(X, y)

    print("\n")
    print("Optimal parameters of linear regression:")
    print(w)

    # Save results in the required format
    print("\n")
    print("Saving the results...")
    np.savetxt("./task1b/results.csv", w, fmt="%.12f")
