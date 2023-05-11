# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

#----  added ------
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("task4/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("task4/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("task4/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("task4/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("task4/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self, input_dim):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dp = nn.Dropout(0.5)
        # self.fc1 = nn.Sequential(
        #     nn.Linear(input_dim, 256),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.5)
        # )

        # self.fc2 = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Dropout(0.5)
        # )

        # self.fc3 = nn.Sequential(
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(64),
        #     nn.Dropout(0.5)
        # )
        
        self.fc4 = nn.Linear(64, 1)
        #---------------


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.dp(torch.relu(self.fc1(x)))
        x = self.dp(torch.relu(self.fc2(x)))
        x = self.dp(torch.relu(self.fc3(x)))
        x = self.fc4(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.fc4(x)
        #-------------------

        return x.view(-1)
    
def make_feature_extractor(x, y, batch_size=256, eval_size=1000):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.

    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net(in_features)
    model.train()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    # Define a list of optimizers and criteria to try
    # optimizer = optim.Adam(model.parameters())
    # criterion = nn.MSELoss()
    
    # # Create data loaders
    # train_data = TensorDataset(x_tr, y_tr)
    # val_data = TensorDataset(x_val, y_val)
    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_data, batch_size=batch_size)

    # # Training loop
    # for epoch in range(10): # for simplicity, we'll just do 100 epochs
    #     print(epoch)
    #     for inputs, targets in train_loader:
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()

    optimizers = [optim.SGD, optim.Adam, optim.Adagrad]
    learning_rates = [0.1, 0.01, 0.001]
    weight_decays = [0.0, 0.01, 0.1]
    criteria = [nn.MSELoss, nn.L1Loss, nn.HuberLoss] 

    best_val_loss = float('inf')
    best_optimizer = None
    best_learning_rate = None
    best_weight_decay = None
    best_criterion = None

    # Iterate over all combinations of optimizers, learning rates, and loss functions
    for optimizer in optimizers:
        print(optimizer)
        for lr in learning_rates:
            for weight_decay in weight_decays:
                for criterion in criteria:
                    # Initialize the optimizer and criterion
                    opt = optimizer(model.parameters(), lr=lr)
                    crit = criterion()

                    # Create data loaders
                    train_data = TensorDataset(x_tr, y_tr)
                    val_data = TensorDataset(x_val, y_val)
                    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_data, batch_size=batch_size)

                    # Training loop
                    for epoch in range(50):  # for simplicity, we'll just do 100 epochs
                        print(epoch)
                        for inputs, targets in train_loader:
                            opt.zero_grad()
                            outputs = model(inputs)
                            loss = crit(outputs, targets)
                            loss.backward()
                            opt.step()

                    # Calculate validation loss
                    val_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            outputs = model(inputs)
                            loss = crit(outputs, targets)
                            val_loss += loss.item() * inputs.size(0)
                    val_loss /= len(val_loader.dataset)

                    # If this is the best validation loss we've seen so far, save these hyperparameters
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_optimizer = optimizer
                        best_learning_rate = lr
                        best_criterion = criterion

    print("Best optimizer:", best_optimizer)
    print("Best learning rate:", best_learning_rate)
    print("Best criterion:", best_criterion)

    # ----------------------


    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        model.eval()
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        if isinstance(x, pd.DataFrame):
            x = x.to_numpy()  # Convert DataFrame to NumPy array
        x_tensor = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float)
            features = model.fc1(x_tensor)
            return features.numpy()
        return x
        # -------------------------

    return make_features

def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()  # Convert DataFrame to NumPy array
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures

def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    # model = LinearRegression()
    # model = RandomForestRegressor(n_estimators=100, random_state=42) #0.28 accuracy
    model = LinearRegression()
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    print('Start pretraining')
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    print('Feature extracted')

    # # regression model
    # regression_model = get_regression_model()

    # y_pred = np.zeros(x_test.shape[0])
    # # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    # # Extract features from training and test data
    # x_train_transformed = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_train)
    # x_test_transformed = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_test)

    # # Train the regression model
    # print('Regression Done')
    # regression_model.fit(x_train_transformed, y_train)

    # # Predict on test set
    # print('Start Prediction')
    # y_pred = regression_model.predict(x_test_transformed)
    # # -------------------------------------

    # assert y_pred.shape == (x_test.shape[0],)
    # y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    # y_pred.to_csv("results.csv", index_label="Id")
    # print("Predictions saved, all done!")

    models = [
    {
        'model': LinearRegression(),
        'params': {}
    },
    {
        'model': Ridge(),
        'params': {
            'alpha': [1, 0.1, 0.01, 0.001, 0.0001]
        }
    },
    {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf']
        }
    },
    {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 10],
        }
    },
    {
        'model': MLPRegressor(random_state=42),
        'params': {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
        }
    }
    ]

    # Create a list to store the results
    results = []

    # Extract features from training data
    x_train_transformed = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_train)

    # Iterate over the models
    for m in models:
        # Perform grid search
        grid = GridSearchCV(m['model'], m['params'], cv=5, scoring='neg_mean_squared_error')
        grid.fit(x_train_transformed, y_train)
        # Append results
        results.append({
            'model': m['model'].__class__.__name__,
            'best_score': grid.best_score_,
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_
        })

    # Sort results by score and print
    results = sorted(results, key=lambda x: x['best_score'], reverse=True)
    for r in results:
        print(f"Model: {r['model']}, Best Score: {r['best_score']}, Best Params: {r['best_params']}")

    # Select the best model
    best_model = results[0]['best_estimator']

    # Extract features from test data
    x_test_transformed = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_test)

    # Predict on test set with the best model
    y_pred = best_model.predict(x_test_transformed)

    # Make sure your prediction array is one-dimensional
    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("task4/results.csv", index_label="Id")
    print("Predictions saved, all done!")