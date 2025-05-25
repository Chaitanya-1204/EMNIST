import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader , Subset , random_split
from torchvision.transforms import functional as F




class RotateTransform:
    
    def __call__(self, x):
        return F.rotate(x , -90)
        

def get_dataloaders(batch_size = 32):
    
    """
        Returns DataLoaders for EMNIST 'byclass' split: train, test, and validation.
    """
    
    transform = transforms.Compose([
        RotateTransform(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307 , ) , (0.3081 , ))
    ])
    
    train_dataset = datasets.EMNIST(
        root = "./data",
        split = "byclass",
        download = True,
        train = True , 
        transform = transform
    )
    
    test_dataset = datasets.EMNIST(
        root = "./data",
        split = "byclass",
        download = True,
        train = False , 
        transform = transform
    )
    
    
    train_data_size = len(train_dataset)
    val_size = train_data_size // 5  # 20% for validation
    train_size = train_data_size - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    
    train_loader = DataLoader(
        train_dataset ,
        batch_size = batch_size , 
        shuffle = True,
        num_workers=8, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset , 
        batch_size  = batch_size, 
        shuffle = False
    )
    
    val_loader = DataLoader(
        val_dataset , 
        batch_size = batch_size ,
        shuffle = False
    )
    
    return train_loader , test_loader , val_loader
    
    
    
    



