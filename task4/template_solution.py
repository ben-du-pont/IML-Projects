# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

# my imports
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV

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
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.dp = nn.Dropout(0.5)
        self.ReLU = nn.ReLU() # add non-linearity to model
        self.sig = nn.Sigmoid()


    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.dp(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.dp(x)
        x = self.fc3(x)
        x = self.sig(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self, in_features):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, in_features),
            nn.ReLU()
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded


class FinalPredictor(nn.Module):
    def __init__(self, encoded_features, y):
        super(FinalPredictor, self).__init__()
        self.predictor = nn.Linear(encoded_features.shape[1], 1)
        self.y = y

    def forward(self, x):
        return self.predictor(x)

    def loss(self, outputs):
        criterion = nn.MSELoss()
        return criterion(outputs, self.y)

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
    # model = Net()
    # model.train()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    # Pretraining - Autoencoder
    autoencoder = AutoEncoder(in_features)
    autoencoder.train()

    epochs = 30
    criterion = nn.MSELoss()
    
    optimizer = optim.SGD(autoencoder.parameters(), lr=1, weight_decay=0.001) # weight decay to prevent overfitting.

    # Training loop - Autoencoder
    for epoch in range(epochs):
        train_loss = 0.0
        # Shuffle the training data at the beginning of each epoch
        permutation = torch.randperm(x_tr.size()[0])
        x_tr = x_tr[permutation]

        for i in range(0, x_tr.size()[0], batch_size):
            optimizer.zero_grad()
            batch_x = x_tr[i:i + batch_size]
            encoded, decoded = autoencoder(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        #train_loss = np.divide(train_loss, x_tr.size()[0])
        # Compute validation loss
        with torch.no_grad():
            _, decoded_val = autoencoder(x_val)
            val_loss = criterion(decoded_val, x_val)
            #val_loss = np.divide(train_loss, x_val.size()[0])

        # Print epoch and validation loss
        print("Autoencoder - Epoch{}/{}: train loss={}, val loss={}".format(epoch+1, epochs, train_loss, val_loss))

    # Extract features using the trained autoencoder
    autoencoder.eval()
    with torch.no_grad():
        encoded_features = autoencoder.encode(torch.tensor(x, dtype=torch.float))

    # Final predictor training
    final_predictor = FinalPredictor(encoded_features, y)
    final_predictor.train()



    def make_features(x):
        """
        This function extracts features from the training and test data, used in the actual pipeline 
        after the pretraining.

        input: x: np.ndarray, the features of the training or test set

        output: features: np.ndarray, the features extracted from the training or test set, propagated
        further in the pipeline
        """
        # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
        final_predictor.eval()
        with torch.no_grad():
            encoded_features = autoencoder.encode(torch.tensor(x, dtype=torch.float))
            features = final_predictor(encoded_features)
        return features.numpy()

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
    PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    
    # regression model
    regression_model = get_regression_model()

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.


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
    x_test_transformed = PretrainedFeatureClass(feature_extractor="pretrain").transform(x_test.to_numpy())

    # Predict on test set with the best model
    y_pred = best_model.predict(x_test_transformed)






    # feature_extractor = PretrainedFeatureClass(feature_extractor="pretrain")
    # x_train_features = feature_extractor.transform(x_train)
    # x_test_features = feature_extractor.transform(x_test.to_numpy())
    
    # Regression training and prediction
    # regression_model.fit(x_train_features, y_train)
    # y_pred = regression_model.predict(x_test_features)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("task4/results.csv", index_label="Id")
    print("Predictions saved, all done!")