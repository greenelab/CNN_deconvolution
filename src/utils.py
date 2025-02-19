''' The Following script contains the classes for models and the funtions needed to 
    tune, train and use the CNN and MLP Models '''
    
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
import torchvision
import torchvision.transforms as transforms

def get_dimensions(dataset):
    ''' Get image dimensions from dataset name'''
    if "MNIST" in dataset:
        image_dim = 28
    elif "PCam" in dataset:
        image_dim = 96
    elif "CIFAR10" in dataset:
        image_dim = 32
    else:
        raise ValueError('Data must be MNIST, FashMNIST, PCam or CIFAR10')
    return image_dim


# Shuffle the rows and columns of an image
def shuffle_image_rows_columns(image, shuffle_order_rows, shuffle_order_columns):
    image = image[shuffle_order_rows, :]  # Shuffle rows
    image = image[:, shuffle_order_columns]  # Shuffle columns
    return image

# Shuffle the rows and columns of an image
def shuffle_image_rows_columns_3CH(image, shuffle_order_rows, shuffle_order_columns):
    image = image[shuffle_order_rows, :, :]  # Shuffle rows
    image = image[:, shuffle_order_columns, :]  # Shuffle columns
    return image

# Generate shuffle orders
def generate_shuffle_orders(size):
    shuffle_order_rows = np.random.permutation(size)  # Shuffle rows
    shuffle_order_columns = np.random.permutation(size)  # Shuffle columns
    return shuffle_order_rows, shuffle_order_columns

class PCamDataset(Dataset):
    """Custom Dataset for PCam data."""
    def __init__(self, data_path, targets_path):
        self.data = self._load_h5(data_path)
        self.targets = self._load_h5(targets_path)
        print(f"Data shape: {self.data.shape}, targets shape: {self.targets.shape}")

    def _load_h5(self, file_path):
        with h5py.File(file_path, 'r') as f:
            data = np.array(f['x'] if 'x' in f else f['y'])
        return data.squeeze()  # Remove all singleton dimensions

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        img = self.data[idx]  # (96, 96, 3)
        label = self.targets[idx]  # 0 or 1
        img = img.astype(np.float32) / 255.0  # Scale to [0, 1]
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return img_tensor, label_tensor

def load_training_data(dataset, batch_size, pcam_data_path=None, val_split=0.2, shuffle_order_rows=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
    ])

    if 'MNIST' in dataset and 'Fash' not in dataset:
        full_train_dataset = torchvision.datasets.MNIST(root="~/datasets/mnist", train=True, download=True, transform=transform)
        image_dim = 28
    elif 'MNIST' in dataset and 'Fash' in dataset:
        full_train_dataset = torchvision.datasets.FashionMNIST(root='~/datasets/fashion_mnist', train=True, download=True, transform=transform)
        image_dim = 28
    elif 'CIFAR10' in dataset:
        full_train_dataset = torchvision.datasets.CIFAR10(root="~/datasets/cifar10", train=True, download=True, transform=transform)
        image_dim = 32
    elif 'PCam' in dataset:
        # Load PCam train dataset
        data_path = "~/datasets/pcam/"
        full_train_dataset = PCamDataset(
            data_path=f"{data_path}camelyonpatch_level_2_split_train_x.h5",
            targets_path=f"{data_path}camelyonpatch_level_2_split_train_y.h5"
        )
        image_dim = 96  # PCam images are 96x96x3
    else:
        raise ValueError('Data must be MNIST, FashMNIST, PCam or CIFAR10')

    shuffle = True if 'shuffle' in dataset else False

    if shuffle:
        # Generate shuffle orders
        if shuffle_order_rows is None:
            shuffle_order_rows, shuffle_order_columns = generate_shuffle_orders(image_dim)
        if 'MNIST' in dataset or 'Fash' in dataset:
            full_train_dataset.data = torch.tensor(
                np.array([
                    shuffle_image_rows_columns(img.numpy(), shuffle_order_rows, shuffle_order_columns)
                    for img in full_train_dataset.data]), dtype=torch.uint8)
        elif 'CIFAR10' in dataset or 'PCam' in dataset:
            full_train_dataset.data = np.array([
                shuffle_image_rows_columns_3CH(img, shuffle_order_rows, shuffle_order_columns)
                for img in full_train_dataset.data])

    # Split training data into train and validation sets
    if isinstance(full_train_dataset, PCamDataset):
        # For PCam, validation is pre-defined
        val_dataset = PCamDataset(
            data_path=f"{data_path}camelyonpatch_level_2_split_valid_x.h5",
            targets_path=f"{data_path}camelyonpatch_level_2_split_valid_y.h5"
        )
    else:
        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        full_train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    shuffle_order_rows = shuffle_order_rows if shuffle else 0
    shuffle_order_columns = shuffle_order_columns if shuffle else 0

    return train_loader, val_loader, shuffle_order_rows, shuffle_order_columns

def load_testing_data(dataset, batch_size, pcam_data_path, shuffle_order_rows=None, shuffle_order_columns=None):

    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to a tensor and scales to [0, 1]
    ])

    # Load the full MNIST dataset
    if 'MNIST' in dataset and 'Fash' not in dataset:
        full_test_dataset = torchvision.datasets.MNIST(root="~/datasets/mnist", train=False, download=True, transform=transform)
        image_dim = 28
    elif 'MNIST' in dataset and 'Fash' in dataset:
        # Load FashionMNIST dataset
        full_test_dataset = torchvision.datasets.FashionMNIST(root='~/datasets/fashion_mnist', train=False, download=True, transform=transform)
        image_dim = 28
    elif 'CIFAR10' in dataset:
        full_test_dataset = torchvision.datasets.CIFAR10(root="~/datasets/cifar10", train=False, download=True, transform=transform)
        image_dim = 32
    
    shuffle = True if 'shuffle' in dataset else False

    if shuffle:
        if shuffle_order_columns is None:
            # Generate shuffle orders
            shuffle_order_rows, shuffle_order_columns = generate_shuffle_orders(image_dim)
        # Apply shuffling to the original dataset's data (convert to NumPy, shuffle, convert back to tensor)
        if 'MNIST' in dataset:
            full_test_dataset.data = torch.tensor(
                np.array([
                    shuffle_image_rows_columns(img.numpy(), shuffle_order_rows, shuffle_order_columns)
                    for img in full_test_dataset.data]), dtype=torch.uint8) # Maintain original dtype
        elif 'CIFAR10' in dataset or 'PCam' in dataset:
            full_test_dataset.data = np.array([
                shuffle_image_rows_columns_3CH(img, shuffle_order_rows, shuffle_order_columns)
                for img in full_test_dataset.data])

    # Create DataLoader
    full_test_loader = DataLoader(full_test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    shuffle_order_rows = shuffle_order_rows if shuffle else 0
    shuffle_order_columns = shuffle_order_columns if shuffle else 0
    
    return full_test_loader, shuffle_order_rows, shuffle_order_columns

# Define the CNN model
class SimpleCNN_3CH(nn.Module):
    def __init__(self, cha_input, cha_hidden, fc_hidden, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, cha_input, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(cha_input)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(cha_input, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(cha_hidden)
        self.conv3 = nn.Conv2d(cha_hidden, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(cha_hidden)
        self.fc1 = nn.Linear(cha_hidden * 4 * 4, fc_hidden)
        self.bn4 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, cha_input, cha_hidden, fc_hidden, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, cha_input, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(cha_input)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(cha_input, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(cha_hidden)
        self.conv3 = nn.Conv2d(cha_hidden, cha_hidden, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(cha_hidden)
        self.fc1 = nn.Linear(cha_hidden * 3 * 3, fc_hidden)
        self.bn4 = nn.BatchNorm1d(fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 10)  # 10 classes 

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))  # Flatten
        x = torch.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, fc1_hidden, fc2_hidden, fc3_hidden):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc1_hidden)
        self.bn1 = nn.BatchNorm1d(fc1_hidden)
        self.fc2 = nn.Linear(fc1_hidden, fc2_hidden)
        self.bn2 = nn.BatchNorm1d(fc2_hidden)
        self.fc3 = nn.Linear(fc2_hidden, fc3_hidden)
        self.bn3 = nn.BatchNorm1d(fc3_hidden)
        self.fc4 = nn.Linear(fc3_hidden, 10)  # 10 classes

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # Output layer
        return x
