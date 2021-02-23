import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

class TransferTrainDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    # want to access already downloaded items w/o loading them all into memory. Do this in __getitem__.

    def __init__(self):  # load pics from memory
        train_set = pd.read_csv("names/train.csv") # change back to ../
        self.pictures = train_set['image_id'].astype('string')
        self.disease_labels = train_set['label']

    # process images here, could augment pictures here to save memory
    def __getitem__(self, index):
        if index*7<self.pictures.size:
            c='A'
        elif index*7<2*self.pictures.size:
            c='B'
        elif index*7<3*self.pictures.size:
            c='C'
        elif index*7<4*self.pictures:
            c='D'
        elif index*7<5*self.pictures:
            c='E'
        elif index*7<6*self.pictures:
            c='F'
        else:
            c='G'
        
        img = Image.open("../augmented_data/" + c+self.pictures[index%self.pictures.size]) # change back to ../ for google colab
        trans = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(384), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img = trans(img) 
        
        disease_of_indexed_picture = self.disease_labels[index-(self.pictures.size*int(index/self.pictures.size))]
        #disease_of_indexed_picture = self.disease_labels[index-(self.pictures.size*int(index/[self.pictures.size]))]

        return img, disease_of_indexed_picture

    def __len__(self):
        return 7*len(self.disease_labels)

class TransferValidationDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    # want to access already downloaded items w/o loading them all into memory. Do this in __getitem__.

    def __init__(self):  # load pics from memory
        train_set = pd.read_csv("names/validation.csv") # change back to ../
        self.pictures = train_set['image_id'].astype('string')
        self.disease_labels = train_set['label']

    # process images here, could augment pictures here to save memory
    def __getitem__(self, index):
        img = Image.open("../train_images/" + self.pictures[index]) # change back to ../ for google colab
        
        trans = transforms.Compose([transforms.Resize(384), transforms.CenterCrop(384), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        img = trans(img) 
        
        disease_of_indexed_picture = self.disease_labels[index]

        return img, disease_of_indexed_picture
    def __len__(self):
        return len(self.disease_labels)