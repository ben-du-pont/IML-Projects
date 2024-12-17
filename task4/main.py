import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA


# Utility to load data
def load_data():
    """
    Loads the data from the CSV files and returns it as numpy arrays.

    Returns:
        x_pretrain: np.ndarray - Features of the pretraining set.
        y_pretrain: np.ndarray - Labels of the pretraining set.
        x_train: np.ndarray - Features of the training set.
        y_train: np.ndarray - Labels of the training set.
        x_test: pd.DataFrame - Features of the test set.
    """
    x_pretrain = pd.read_csv("task4/pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("task4/pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("task4/train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("task4/train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("task4/test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_test


# Define a simple MLP regressor
class MLPRegressor(nn.Module):
    """
    A simple feedforward network with 3 layers, used for pretraining.

    Attributes:
        encoder: nn.Sequential - The feature extraction layers.
        predictor: nn.Sequential - The final prediction layer.
    """
    def __init__(self, *, input_size, layers, activation):
        super(MLPRegressor, self).__init__()
        all_layers = [input_size, *layers]
        self.encoder = nn.Sequential(
            *[
                layer
                for i in range(len(all_layers) - 1)
                for layer in [
                    nn.Linear(all_layers[i], all_layers[i + 1]),
                    activation() if i < len(all_layers) - 2 else nn.Identity(),
                ]
            ]
        )
        self.predictor = nn.Sequential(
            nn.ReLU(), nn.Linear(all_layers[-1], 1)
        )

    def forward(self, x):
        return self.predictor(self.encoder(x)).squeeze(-1)

    def make_features(self, _x):
        """
        Extracts the encoded features for downstream tasks.

        Args:
            _x: np.ndarray - Input data for feature extraction.

        Returns:
            torch.Tensor - Extracted features.
        """
        self.eval()
        with torch.no_grad():
            return self.encoder(torch.tensor(_x, dtype=torch.float))


def pretrain_model(x, y, batch_size=200, eval_size=1000, epochs=100, learning_rate=0.001):
    """
    Trains the MLPRegressor model on the pretraining data.

    Args:
        x: np.ndarray - Features for pretraining.
        y: np.ndarray - Labels for pretraining.
        batch_size: int - Batch size for training.
        eval_size: int - Size of validation set.
        epochs: int - Number of training epochs.
        learning_rate: float - Learning rate for optimizer.

    Returns:
        MLPRegressor - Trained model.
    """
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=eval_size, random_state=42, shuffle=True)
    x_train, x_val = torch.tensor(x_train, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_train, y_val = torch.tensor(y_train, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    model = MLPRegressor(input_size=x_train.shape[1], layers=[256, 64], activation=nn.ReLU)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(x_train.size(0))
        epoch_loss = 0.0

        for i in range(0, x_train.size(0), batch_size):
            optimizer.zero_grad()
            idx = perm[i:i + batch_size]
            x_batch, y_batch = x_train[idx], y_train[idx]
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            y_val_pred = model(x_val)
            val_loss = loss_fn(y_val_pred, y_val).item()

        print(f"Epoch {epoch + 1}/{epochs}: Training Loss = {epoch_loss:.4f}, Validation Loss = {val_loss:.4f}")

    return model


def main():
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()

    # Pretraining the feature extractor
    print("Pretraining the feature extractor...")
    feature_extractor = pretrain_model(x_pretrain, y_pretrain)

    # Feature extraction
    print("Extracting features...")
    x_train_features = feature_extractor.make_features(x_train).numpy()
    x_test_features = feature_extractor.make_features(x_test.to_numpy()).numpy()

    # Data normalization
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_features)
    x_test_scaled = scaler.transform(x_test_features)

    # Train regression model
    print("Training Ridge regression...")
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0]).fit(x_train_scaled, y_train)

    # Predictions
    print("Generating predictions...")
    y_pred = ridge.predict(x_test_scaled)
    y_pred_df = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred_df.to_csv("task4/results.csv", index_label="Id")

    print("Predictions saved to task4/results.csv!")


if __name__ == "__main__":
    main()
