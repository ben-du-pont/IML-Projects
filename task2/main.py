# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = np.zeros_like(train_df['price_CHF'])
    X_test = np.zeros_like(test_df)

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

# ------------------------------------

    # Remove all rows where the subset 'price_CHF' has missing values (NaN)
    train_df = train_df.dropna(subset=['price_CHF'])

    # Convert the columns 'season' into a binary value (0 or 1)
    train_df = pd.get_dummies(train_df, columns=['season'])
    test_df = pd.get_dummies(test_df, columns=['season'])

    # Price_CHF is the target so must be drop from X_train and initialized as y_train
    X_train = train_df.drop(['price_CHF'], axis=1).values
    y_train = train_df['price_CHF'].values


    # Initialize X_test
    X_test = test_df.values

    # Replace missing values by the mean of each column of X_train
    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
    X_train = imputer.fit(X_train).transform(X_train)
    X_test = imputer.transform(X_test)

#--------------------------------------

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
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions

# -----------------------------------------

    # # LINER MODEL 
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)

    # # RIDGE MODEL
    # model = Ridge(0.1, fit_intercept = False, positive=False)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)


    # CROSS-VALIDATION
    kf_splits = 10
    alpha_range = np.logspace(0.0000001, 5.0, num=2) # tried with 100 and still returned alpha = 1 as the best 

    kf = KFold(n_splits=kf_splits, shuffle=True, random_state=1) # Shuffle=True is essential for higher performance
    #Result with Shuffle=False is 0.78 instrad of 0.93

    scores = []
    models = []

    for alpha in alpha_range:
        # RIDGE
        # model = Ridge(alpha=alpha,fit_intercept = True, tol=1e-7, random_state=1, solver='sparse_cg')

        # KERNEL (Radial Basis Function)
        model = KernelRidge(alpha=alpha, kernel="rbf")

        # LINEAR
        # model = LinearRegression()

        model_scores = []
        for train_idx, val_idx in kf.split(X_train):

            kf_X_train, X_val = X_train[train_idx], X_train[val_idx]
            kf_y_train, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(kf_X_train, kf_y_train)
            model_scores.append(model.score(X_val, y_val))

        scores.append(np.mean(model_scores))
        models.append(model)

    # Keep the best model
    best_model_idx = np.argmax(scores)
    best_model = models[best_model_idx]


    print("Best alpha:", alpha_range[best_model_idx])
    print("Best score:", scores[best_model_idx])
    
    # Use the best model for prediction
    y_pred = best_model.predict(X_test)

# ------------------------------------------
    

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred


# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('task2/results.csv', index=False)
    print("\nResults file successfully generated!")

