import torch.nn as nn 
from data import get_dataloaders
from model import CNN , count_params , train, eval
import torch


# Set the device
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using Device {device}")

# get Dataloaders
train_loader , test_loader , val_loader = get_dataloaders(32)


# Getting model 

model = CNN()
model = model.to(device)

print(f"Model Parameters : {count_params(model):,}")



# Define Training Loop 

num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters() , lr = 0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    
    train(model , train_loader , optimizer , criterion , device , epoch)
    eval(model , criterion , val_loader , device)
    
print("Testing .........")
eval(model , criterion , test_loader , device)


