import torch
import torch.nn as nn
import torch.nn.functional as F


#Gets updated output dimension for Conv2d or MaxPool2d Layer
#Taken from https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html and https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
def getUpdatedDimension(input_dimension,padding,dilation,kernel,stride):
    x=input_dimension+2*padding-dilation*(kernel-1)-1
    return int((x/stride)+1)

class StartingNetwork(nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """
    def __init__(self,input_channels, input_width,input_height,output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)

        input_height=getUpdatedDimension(input_height,0,1,5,1)
        input_width=getUpdatedDimension(input_width,0,1,5,1)
        input_height=getUpdatedDimension(input_height,0,1,2,2)
        input_width=getUpdatedDimension(input_width,0,1,2,2)
        
        self.conv2 = nn.Conv2d(6, 16, 5)
        input_height=getUpdatedDimension(input_height,0,1,5,1)
        input_width=getUpdatedDimension(input_width,0,1,5,1)
        input_height=getUpdatedDimension(input_height,0,1,2,2)
        input_width=getUpdatedDimension(input_width,0,1,2,2)
        #print(input_width)
        #print(input_height)
        self.flattened_dim=16*input_width*input_height
        self.fc1 = nn.Linear(self.flattened_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print(x.shape) # torch.Size([32, 16, 147, 197])
        x = torch.reshape(x, (32,self.flattened_dim))
        print(x.shape)
        x = self.fc1(x)
        print(x.shape)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        print(x.shape)
        return x
