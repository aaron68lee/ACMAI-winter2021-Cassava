import torch
import torchvision.transforms as transforms
import matplotlib.image as mpimg
import pandas as pd


class OurDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """
    # want to access already downloaded items w/o loading them all into memory. Do this in __getitem__.

    def __init__(self):  # load pics from memory
        train_set = pd.read_csv("data/train.csv")
        self.pictures = train_set['image_id'].astype('string')
        self.disease_labels = train_set['label'].astype('string')

    # process images here, could augment pictures here to save memory
    def __getitem__(self, index):
        img = mpimg.imread("data/train_images/"+self.pictures[index])
        trans = transforms.ToTensor()
        img=trans(img) 
        
        disease_of_indexed_picture = self.disease_labels[index]

        return img, disease_of_indexed_picture

    def __len__(self):
        return len(self.disease_labels)