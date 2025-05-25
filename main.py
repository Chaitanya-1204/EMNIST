import torch 
from data import get_dataloaders


# Set the device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

# get Dataloaders
train_loader , test_loader , val_loader = get_dataloaders(32)

