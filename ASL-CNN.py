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
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.conv4 = nn.Conv2d(64, 128, 5)

        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, len(classes))

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

# training function
def train(model, dataloader):
    print('Training')
    model.train()
    running_loss = 0.0
    running_correct = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, target = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_function(outputs, target)
        running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        # print(preds)
        running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        
    train_loss = running_loss/len(dataloader.dataset)
    train_accuracy = 100. * running_correct/len(dataloader.dataset)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}")
    
    return train_loss, train_accuracy
    
train_transform=transforms.Compose([transforms.ColorJitter(brightness=0.3, 
                          saturation=0.1, contrast=0.1),transforms.ToTensor()])
Train_data_path="ASL/asl_alphabet_train/asl_alphabet_train"
train_data=ImageFolder(Train_data_path,transform=train_transform)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True) #batch size of 100
classes=train_loader.dataset.classes

Model_1=CustomCNN()

loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(Model_1.parameters(),lr=0.001,weight_decay=0.0001)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("The device in use is {}".format(device))
Model_1.to(device)

train_loss , train_accuracy = [], []
val_loss , val_accuracy = [], []S

Num=10
for epoch in range(Num):
    print(f"Epoch {epoch+1} of {Num}")
    train_epoch_loss, train_epoch_accuracy = train(Model_1, train_loader)
    train_loss.append(train_epoch_loss)
    train_accuracy.append(train_epoch_accuracy)

# accuracy plots
plt.figure(figsize=(10, 7))
plt.plot(train_accuracy, color='green', label='train accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('outputs/accuracy.png')
plt.show()
 
# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('outputs/loss.png')
plt.show()

# save the model to disk
print('Saving model...')
torch.save(Model_1.state_dict(), 'outputs/model.pth')