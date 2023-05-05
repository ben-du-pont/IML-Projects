# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet50_Weights, resnet50
from torch import optim

import matplotlib.pyplot as plt

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # MATEO (WINDOWS)
# device = torch.device("cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # BEN (MAC OS)


def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract 
    the embeddings.
    """
    # TODO: define a transform to pre-process the images
    train_transforms = transforms.Compose([
    transforms.Resize((200,400)),  # Resize the image to 256x256 pixels
    #transforms.CenterCrop(200),  # Crop the image to 224x224 pixels around the center
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    train_dataset = datasets.ImageFolder(root="task3/dataset/", transform=train_transforms)
    # Hint: adjust batch_size and num_workers to your PC configuration, so that you don't 
    # run out of memory
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False,
                              pin_memory=True, num_workers=8)

    # TODO: define a model for extraction of the embeddings (Hint: load a pretrained model,
    #  more info here: https://pytorch.org/vision/stable/models.html)

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()  # Set the model to evaluation mode

    


    num_images = len(train_dataset)
    # embeddings = np.zeros((num_images, embedding_size))

    # TODO: Use the model to extract the embeddings. Hint: remove the last layers of the 
    # model to access the embeddings the model generates. 
    modules = list(model.children())[:-1]
    model = torch.nn.Sequential(*modules)
    model.to(device)

    # i = 0
    # for images, _ in train_loader:

    #     print(np.transpose(images[0], (1,2,0)).shape)
    #     print(images[i])
    #     plt.imshow((images[i].detach().numpy().transpose(1, 2, 0)*255).astype(np.uint8))
    #     plt.show()
    #     i = i + 1
    #     break
    
    embeddings = []
    iter = 0
    print(iter)
    for images, _ in train_loader:
        print(iter)
        with torch.no_grad():
            features = model(images)
            embeddings.append(features)
        iter += 1
    embeddings = torch.cat(embeddings, dim=0)

    np.save('task3/dataset/embeddings.npy', embeddings)


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
    # TODO: Normalize the embeddings across the dataset

    # norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # embeddings = embeddings / norm

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

# Hint: adjust batch_size and num_workers to your PC configuration, so that you don't run out of memory
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

# TODO: define a model. Here, the basic structure is defined, but you need to fill in the details
class Net(nn.Module):
    """
    The model class, which defines our classifier.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # Define the convolutional layers
        self.fc1 = nn.Linear(3000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)

        return x

def train_model(train_loader):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.

    input: train_loader: torch.data.util.DataLoader, the object containing the training data
    
    output: model: torch.nn.Module, the trained model
    """
    model = Net()
    model.train()
    model.to(device)
    n_epochs = 10
    # TODO: define a loss function, optimizer and proceed with training. Hint: use the part 
    # of the training data as a validation split. After each epoch, compute the loss on the 
    # validation split and print it out. This enables you to see how your model is performing 
    # on the validation data before submitting the results on the server. After choosing the 
    # best model, train it on the whole training data.
    
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()

    # TODO: proceed with training
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for [X, y] in train_loader:
            
            optimizer.zero_grad()
            y_pred = model(X)
            loss = criterion(y_pred, y.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * X.shape[0]
        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1}: loss={epoch_loss:.4f}")

    # for epoch in range(n_epochs):

    #     for [X, y] in train_loader:
    #         pass
    return model

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
