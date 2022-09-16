import torch
from torch import nn
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from CustomImageDataset import CustomImageDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(opt):
    train_dataset = CustomImageDataset('train_labels.csv','environment/TextRecognitionDataGenerator/trdg/Training/')
    train_dataloader = DataLoader(train_dataset,batch_size=256,shuffle=True)

