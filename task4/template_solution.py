import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
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

class AutoNN(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        self.l1 = nn.Linear(1000, 400)
        self.l2 = nn.Linear(400, 100)
        self.l3 = nn.Linear(100, 40)
        self.l4 = nn.Linear(40, 1)

    def reduce(self, x):
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.l2(x)
        return x

    def forward(self, x):
        x = self.l1(x)
        x = nn.functional.relu(x)
        x = self.l2(x)
        x = nn.functional.sigmoid(x)
        x = self.l3(x)
        x = nn.functional.relu(x)
        x = self.l4(x)

        return x

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
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float).squeeze(), torch.tensor(x_val, dtype=torch.float).squeeze()
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # Model declaration
    model = AutoNN()
    model.train()

    # Choose SGD as optimiser and MSE as loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    # Set the number of epochs
    n_epochs = 150

    num_points = x_tr.size(0)
    prev_val_loss = 100.0

    for curr_epoch in range(n_epochs):
        # Go to train mode
        model.train()
        # Get permutation of the dataset
        random_permutation = torch.randperm(num_points)
        for k in range(0, num_points, batch_size):
            optimizer.zero_grad()

            data_indices = random_permutation[k:k + batch_size]
            x_tr_batch = x_tr[data_indices]
            y_tr_batch = y_tr[data_indices]

            # Forward and backward steps
            x_tr_pred = model(x_tr_batch)
            loss = loss_fn(x_tr_pred.squeeze(), y_tr_batch)
            loss.backward()
            optimizer.step()

         # Go to evaluation mode and perform validation step
        model.eval()
        x_val_pred = model(x_val)
        val_loss = loss_fn(x_val_pred.squeeze(), y_val)
        print(f'Epoch {curr_epoch + 1}|{n_epochs}. Validation loss: {val_loss.item()}')

        # Stop training if validation loss is higher than in previous iteration
        if(val_loss.item() > prev_val_loss):
            print(f'Validation loss in iteration {curr_epoch + 1} was higher than in previous. Stopping training.')
            break
        else:
            prev_val_loss = val_loss.item()

    # Go to evaluation model before returning the model
    model.eval()

    return model


def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    model = RidgeCV(alphas=[0.05, 0.1, 0.3, 0.5, 1.0])
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()
    print("Data loaded!")
    # Utilize pretraining data by creating feature extractor which extracts lumo energy
    # features from available initial features
    feature_extractor =  make_feature_extractor(x_pretrain, y_pretrain)

    # regression model
    regression_model = get_regression_model()

    # Encode both train & test datasets to a smaller feature set + add LEMO prediction as feature
    x_train_reduced = np.concatenate((feature_extractor.reduce(torch.tensor(x_train, dtype=torch.float)).detach().numpy(), feature_extractor(torch.tensor(x_train, dtype=torch.float)).detach().numpy()), axis=1)
    x_test_reduced = np.concatenate((feature_extractor.reduce(torch.tensor(x_test.to_numpy(), dtype=torch.float)).detach().numpy(), feature_extractor(torch.tensor(x_test.to_numpy(), dtype=torch.float)).detach().numpy()), axis=1)

    # Train regression on the encoded train dataset & make prediction on encoded test dataset
    regression = regression_model.fit(x_train_reduced, y_train)
    y_pred = regression.predict(x_test_reduced)

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("task4/results.csv", index_label="Id")
    print("Predictions saved, all done!")
