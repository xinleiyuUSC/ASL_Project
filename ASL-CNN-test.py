# -*- coding: utf-8 -*-

from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.nn.modules.dropout import Dropout
import torchvision
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 29)

        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
test_filepath = "ASL/asl_alphabet_test/"
test_transforms = transforms.Compose([transforms.ToTensor()])

test_dataset = torchvision.datasets.ImageFolder(test_filepath, transform=test_transforms)
print("Test Dataset Info:\n",test_dataset)

test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=1) #only 1 img for 1 class


test_filepath = "ASL/asl_alphabet_test/asl_alphabet_test/"
labels_map = {'A':0,'B':1,'C': 2, 'D': 3, 'E':4,'F':5,'G':6, 'H': 7, 'I':8, 'J':9,'K':10,'L':11, 'M': 12, 'N': 13, 'O':14, 
                'P':15,'Q':16, 'R': 17, 'S': 18, 'T':19, 'U':20,'V':21, 'W': 22, 'X': 23, 'Y':24, 'Z':25, 
                'del': 26, 'nothing': 27,'space':28}

test_labels = []
for folder_name in os.listdir(test_filepath):
    label = folder_name.replace("_test.jpg","")
    label = labels_map[label]
    test_labels.append(np.array(label))
test_labels.sort()

device = "cuda" if torch.cuda.is_available() else "cpu"
Model_1 =CustomCNN()
Model_1.load_state_dict(torch.load('outputs/model.pth'))
pred_test=[]
test_labels_list=[]
with torch.no_grad():
    correct = 0
    for (images,_),labels in zip(test_dataloader,test_labels):
        Model_1.eval()
        images = images.to(device)
        output = Model_1(images)
        prediction = torch.max(output,1)[1]
        pred_test.append(prediction.cpu().numpy()[0])
        correct += (prediction.cpu().numpy()[0] == labels)
        test_labels_list.append(labels)
    print("Accuracy :",(correct/len(test_dataloader.dataset))*100,"%")
