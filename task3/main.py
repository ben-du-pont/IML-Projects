import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.inception import inception_v3, Inception_V3_Weights
from torch import optim
from torch.utils.data import random_split, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from PIL import Image

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # (WINDOWS/LINUX)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # (MAC OS)

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # Define a transform to pre-process the images
    mean = [0.485, 0.456, 0.406] # from ImageNet dataset
    std = [0.229, 0.224, 0.225] # from ImageNet dataset

    train_transforms = transforms.Compose([
    transforms.Resize((256,256)),  # resizes to a 256x256 image
    transforms.ToTensor(),  # converts to a Pytorch tensor
    transforms.Normalize(mean=mean, std=std) # normalizes the tensor
    ])

    train_dataset = datasets.ImageFolder(root="task3/dataset/", transform=train_transforms)

    # If needed, adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True, num_workers=8)

    # Extract the embeddings from the imagees using a pretrained model, ResNet50
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    modules = list(resnet.children())[:-1] # takes all layers except the last one
    resnet = nn.Sequential(*modules)
    embeddings = []
    count = 0
    for img, _ in train_loader:
        count += batch_size
        print(count)
        output = resnet(img.to(device)).cpu().detach().squeeze().numpy()
        embeddings.append(output)

    embeddings = np.vstack(embeddings)


    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # normalize the embeddings

    print("Embeddings generated")

    np.save('task3/dataset/embeddings.npy', embeddings)
    print("Embeddings saved to embeddings.npy")


def get_data(file, train=True):
    """
    Load the triplets from the file and generate the features and labels.

    input: file: string, the path to the file containing the triplets
          train: boolean, whether the data is for training or testing

    output: X: numpy array, the features
            y: numpy array, the labels
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line)

    # generate training data from triplets
    train_dataset = datasets.ImageFolder(root="task3/dataset/",
                                         transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('task3/dataset/embeddings.npy')

    file_to_embedding = {}
    for i in range(len(filenames)):
        file_to_embedding[filenames[i]] = embeddings[i]
    X = []
    y = []
    # use the individual embeddings to generate the features and labels for triplets
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))
        y.append(1)
        # Generating negative samples (data augmentation)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))
            y.append(0)
    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

# Adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
def create_loader_from_np(X, y = None, train = True, batch_size=64, shuffle=True, num_workers = 4):
    """
    Create a torch.utils.data.DataLoader object from numpy arrays containing the data.

    input: X: numpy array, the features
           y: numpy array, the labels
    
    output: loader: torch.data.util.DataLoader, the object containing the data
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float), 
                                torch.from_numpy(y).type(torch.long))
    else:
        dataset = TensorDataset(torch.from_numpy(X).type(torch.float))
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        pin_memory=True, num_workers=num_workers)
    return loader


class Net(nn.Module): 
    """
    The model class, which defines our classifier.
    """
    def __init__(self, d):
        """
        The constructor of the model.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3 * d, 1024),  # Input layer (3 * d features -> 128)
            nn.Dropout(0.7),         # Dropout layer with 70% probability
            nn.BatchNorm1d(1024),     # Batch normalization
            nn.ReLU(),               # ReLU activation
            
            nn.Linear(1024, 128),      # Hidden layer (128 -> 64)
            nn.Dropout(0.5),         # Dropout layer with 50% probability
            nn.BatchNorm1d(128),      # Batch normalization
            nn.ReLU(),               # ReLU activation
            
            nn.Linear(128, 1),         # Output layer (64 -> 1)

            nn.Sigmoid()             # Sigmoid activation for binary classification
        )

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        return self.model(x)

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """


    dataset_size = len(train_loader.dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_set, val_set = random_split(train_loader.dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)

    
    d = 2048  # Number of features in resnet50 embeddings
    learning_rate = 1e-3

    model = Net(d)
    model.train()
    model.to(device)
    num_epochs = 50

    # Initialize model, loss function, and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
    sched = ReduceLROnPlateau(optimizer, patience=3) # adapt learning rate to validation loss evolution

    # Lists to store losses for visualization
    train_losses = []
    val_losses = []

    val_loss_best = np.inf
    idx = 0
    idx_stop = 8

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}')
        model.train()  # Set the model to training mode
        epoch_train_loss = 0.0
        
        # Iterate over the batches in the training loader
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)

            # Compute the loss
            loss = criterion(outputs, y_batch.float().unsqueeze(1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Average training loss for this epoch
        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        
        # Validation phase
        model.eval()  # Set the model to evaluation mode
        epoch_val_loss = 0.0
        
        with torch.no_grad():  # No need to compute gradients during validation
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(x_batch)
                
                # Compute the loss
                loss = criterion(outputs,  y_batch.float().unsqueeze(1))
                epoch_val_loss += loss.item()
        
        # Average validation loss for this epoch
        epoch_val_loss /= len(val_loader)
        sched.step(epoch_val_loss)
        val_losses.append(epoch_val_loss)


        
        print(f'Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}')
        # Print accuracy every 10 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}, Train Accuracy: {compute_accuracy(model, train_loader)}')
            print(f'Epoch {epoch+1}, Validation Accuracy: {compute_accuracy(model, val_loader)}')

        val_loss_best, idx = (epoch_val_loss, 0) if epoch_val_loss < val_loss_best else (val_loss_best, idx + 1)
        if idx >= idx_stop:
            print("Early stopping")
            break
    # Plotting the loss curves
    plt.plot(range(1, epoch+2), train_losses, label='Training Loss')
    plt.plot(range(1, epoch+2), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


    return model

def compute_accuracy(model, loader):
    """
    This function computes the accuracy of the model on the provided data loader.

    input: 
        model: torch.nn.Module, the trained model
        loader: torch.utils.data.DataLoader, the data loader containing the features and labels

    output:
        accuracy: float, the accuracy on the given dataset
    """
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No need to compute gradients
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            # Forward pass: Get predictions from the model
            outputs = model(x_batch)
            predicted = (outputs >= 0.5).float()  # Threshold at 0.5 for binary classification

            # Update total and correct counts
            total += y_batch.size(0)
            correct += (predicted.view(-1) == y_batch).sum().item()

    accuracy = correct / total
    return accuracy



def test_model(model, loader):
    """
    The testing procedure of the model; it accepts the testing data and the trained model and 
    then tests the model on it.

    input: model: torch.nn.Module, the trained model
           loader: torch.data.util.DataLoader, the object containing the testing data
        
    output: None, the function saves the predictions to a results.txt file
    """
    model.eval()
    predictions = []
    # Iterate over the test data
    with torch.no_grad(): # We don't need to compute gradients for testing
        for [x_batch] in loader:
            x_batch= x_batch.to(device)
            predicted = model(x_batch)
            predicted = predicted.cpu().numpy()
            # Rounding the predictions to 0 or 1
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)
        predictions = np.vstack(predictions)
    np.savetxt("task3/results.txt", predictions, fmt='%i')


# Main function. You don't have to change this
if __name__ == '__main__':
    TRAIN_TRIPLETS = 'task3/train_triplets.txt'
    TEST_TRIPLETS = 'task3/test_triplets.txt'

    # generate embedding for each image in the dataset
    if(os.path.exists('task3/dataset/embeddings.npy') == False):
        generate_embeddings()

    # load the training and testing data
    X, y = get_data(TRAIN_TRIPLETS)
    X_test, _ = get_data(TEST_TRIPLETS, train=False)

    # Create data loaders for the training and testing data
    train_loader = create_loader_from_np(X, y, train = True, batch_size=64)
    test_loader = create_loader_from_np(X_test, train = False, batch_size=2048, shuffle=False)

    # define a model and train it
    model = train_model(train_loader)

    # test the model on the test data
    test_model(model, test_loader)
    print("Results saved to results.txt")
