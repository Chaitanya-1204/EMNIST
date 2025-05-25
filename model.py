import torch 
import torch.nn as nn
from tqdm import tqdm

class CNN(nn.Module):
    
    def __init__(self):
        
        super(CNN , self).__init__()
        
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(1 , 32 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.Conv2d(32 , 32 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
        
            nn.Conv2d(32 , 64 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64 , 64 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(64 , 128 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128 , 128 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.MaxPool2d(2),
            
            nn.Conv2d(128 , 256 , kernel_size =  3 , padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fcn = nn.Linear(256 , 62)
        
        
        
    def forward(self , x):
        x = self.conv_layers(x)
        x = x.view(x.size(0) , -1)
        x = self.fcn(x)
        return x
        


def count_params(model):
    return sum(p.numel() for p in model.parameters() )



def train(model , dataloader , optimizer , criterion , device , epoch):
    
    total_loss = 0.0
    correct = 0 
    total = 0 
    
    
    for images , labels in tqdm(dataloader , desc = f"Epoch : {epoch + 1}"):
        
        # move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images) # forward pass
        loss = criterion(outputs , labels) 
        
        # update weights
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Stats 
        
        _ , pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        
        total += images.size(0)
        total_loss += loss.item() * images.size(0)
        
    epoch_loss = total_loss / total
    accuracy = correct / total
    
    print(f"Epoch {epoch + 1 } | Loss : {epoch_loss} | Accuracy : {accuracy}")
    
        
def eval(model , criterion , dataloader , device):
    
    total_loss = 0.0
    correct = 0 
    total = 0 
    
    
    for images , labels in tqdm(dataloader , desc = f"Evaluating : "):
        
        # move data to device
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images) # forward pass
        loss = criterion(outputs , labels) 
     
        
        # Stats 
        
        _ , pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        
        total += images.size(0)
        total_loss += loss.item() * images.size(0)
        
    epoch_loss = total_loss / total
    accuracy = correct / total
    
    print(f"Loss : {epoch_loss} | Accuracy : {accuracy}")
    
        
        
          
        