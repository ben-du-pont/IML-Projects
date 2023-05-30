import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
#from sklearn.base import BaseEstimator, TransformerMixin
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import RidgeCV


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
        self.fc1 = nn.Linear(1000, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def encoder(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """

        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sig(x)
        x = self.fc3(x)
        x = self.relu(x)
        return self.fc4(x)

def make_feature_extractor(x, y, batch_size=200, eval_size=1000):
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
    # in_features = x.shape[-1]
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float).squeeze(), torch.tensor(x_val, dtype=torch.float).squeeze()
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # Model declaration
    model = Net()
    model.train()

    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)

    # Choosing Optimizer and Loss function
    opti = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fun = nn.MSELoss()

    n_epochs = 50
    best_val_loss = np.inf

    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:

            opti.zero_grad()

            loss = loss_fun(model(x_batch), y_batch.float().unsqueeze(1))
            loss.backward()
            opti.step()

        # Start evaluation
        model.eval()
        val_loss = loss_fun(model(x_val), y_val.float().unsqueeze(1))
        print(f'epoch {epoch+1} --- validation loss: {val_loss.item()}')

        # Early Stopping to prevent overfitting
        if(val_loss.item() > best_val_loss):
            print('Early stopping...')
            break
        else:
            best_val_loss = val_loss.item()

    return model


def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.

    # Ridge including cross-validation & regularization (prevent overfitting)
    alphas = [0.01, 0.1, 0.25, 0.5, 0.75, 1.0]
    model = RidgeCV(alphas=alphas)
    # ---------------------------------

    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)
    #PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})

    # regression model
    regression_model = get_regression_model()

    #y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.

    with torch.no_grad():
        # Create features from Train dataset + add LUMO prediction from pre-training    
        x_train_features = feature_extractor.encoder(torch.tensor(x_train, dtype=torch.float)).numpy()
        train_features_LEMO = feature_extractor(torch.tensor(x_train, dtype=torch.float)).numpy()

        # Create features from Test dataset + add LUMO prediction from pre-training
        x_test_features = feature_extractor.encoder(torch.tensor(x_test.to_numpy(), dtype=torch.float)).numpy()
        test_features_LEMO = feature_extractor(torch.tensor(x_test.to_numpy(), dtype=torch.float)).numpy()

    # Create new encoded Train & Test dataset
    x_train_encoded = np.concatenate((x_train_features, train_features_LEMO), axis=1)
    x_test_encoded = np.concatenate((x_test_features, test_features_LEMO), axis=1)

    # Regression on encoded dataset
    model = regression_model.fit(x_train_encoded, y_train)
    y_pred = model.predict(x_test_encoded)
    # ----------------------------------

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("task4/results.csv", index_label="Id")
    print("Predictions saved, all done!")
