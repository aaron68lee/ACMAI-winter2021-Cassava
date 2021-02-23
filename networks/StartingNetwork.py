import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
        
        ''' our convolutional layers are in this comment block
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
        
        
        #add more conv layers and max pool after
        
        #self.conv3 = nn.Conv2d(6, 16, 5)
        input_height=getUpdatedDimension(input_height,0,1,5,1)
        input_width=getUpdatedDimension(input_width,0,1,5,1)
        input_height=getUpdatedDimension(input_height,0,1,2,2)
        input_width=getUpdatedDimension(input_width,0,1,2,2)

        resnet = models.resnet18(pretrained = True)
        resnet = nn.Sequential(*list(model.children())[:-1])
        print(resnet)
        #resnet.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
        resnet.eval()

        '''
        self.resnet = models.resnet50(pretrained = True)
        #self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.resnet.eval()
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        
        self.flattened_dim=16*input_width*input_height
        
        self.fc=nn.Linear(2048,512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_dim)
        

    def forward(self, x):
        '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.reshape(x, (x.size()[0],self.flattened_dim))
        '''

        # Call self.resnet here
        # Freeze gradients with eval() or sumn
        # Remove last layer, and add new one
        #https://pytorch.org/vision/stable/models.html 

        with torch.no_grad():
            x = self.resnet(x)
        
        for name, child in self.resnet:
            print(name)
        print('\n\n\n')
        
        x = torch.reshape(x, (x.size()[0],2048))

        x=self.fc(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
