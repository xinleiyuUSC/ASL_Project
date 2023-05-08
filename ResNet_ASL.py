import os
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn.metrics import f1_score, confusion_matrix
import seaborn as sns
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split, DataLoader
import torchvision.transforms as transforms
from torch.nn.modules.dropout import Dropout
import time

def resNet34():
    train_transform=transforms.Compose([transforms.ColorJitter(brightness=0.3, 
                            saturation=0.1, contrast=0.1),transforms.ToTensor()])

    Train_data_path="./asl_alphabet_train/asl_alphabet_train/"
    train_data=ImageFolder(Train_data_path,transform=train_transform)
    print(len(train_data))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The device in use is {}".format(device)) 

    test_filepath = "./asl_alphabet_test/asl_alphabet_test/"
    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    test_dataset = torchvision.datasets.ImageFolder(test_filepath, transform=test_transforms)
    print("Test Dataset Info:\n",test_dataset)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1)
    
 
    labels_map = {'A':0,'B':1,'C': 2, 'D': 3, 'E':4,'F':5,'G':6, 'H': 7, 'I':8, 'J':9,'K':10,'L':11, 'M': 12, 'N': 13, 'O':14, 
                    'P':15,'Q':16, 'R': 17, 'S': 18, 'T':19, 'U':20,'V':21, 'W': 22, 'X': 23, 'Y':24, 'Z':25, 
                    'del': 26, 'nothing': 27,'space':28}
    test_labels = []
    for folder_name in os.listdir(test_filepath):
        label = folder_name.replace("_test.jpg","")
        label = labels_map[label]
        test_labels.append(np.array(label))
    test_labels.sort()
        
    def train_data_loader(data_train, batchsize):
        train_loader=DataLoader(dataset=data_train, 
                                batch_size=batchsize,
                                shuffle=True, num_workers=2)
        return train_loader

    def Model_train(number_of_epochs, train_loader, Model, loss_function, optimizer):
        count=0
        training_accuracy = []
        training_lost = []
        for epoch in range(number_of_epochs):
            correct=0
            for images, labels in train_loader:
                count+=1
                images = images.cuda()
                labels = labels.cuda()
                outputs=Model(images)
                loss=loss_function(outputs, labels)
                # Back Propogation
                optimizer.zero_grad()
                loss.backward()
                # Update Weights (Optimize the model)
                optimizer.step()
                # Checking the performance
                predictions=torch.max(outputs,1)[1]
                correct+=(predictions==labels).cpu().sum().numpy()
                
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            training_accuracy.append(100*correct/len(train_loader.dataset))
            loss_string = str(loss.data)
            #training_lost.append(loss_string)
            print(current_time)
            print("Epoch is: {0}, Loss is {1} and Accuracy is: {2}".format(epoch+1,loss.data,100*correct/len(train_loader.dataset)))

        print("Training finished")
        
        plt.title("Training Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracu  (%)")
        plt.plot(range(1, number_of_epochs +1), np.array(training_accuracy), label = "Accuracy during the training" )
        plt.legend()
        plt.show()
        
        print("start to test")
    
        pred_test=[]
        test_labels_list=[]
        with torch.no_grad():
            correct = 0
            for (images,_),labels in zip(test_dataloader,test_labels):
                resnet_model34.eval()
                images = images.to(device)
                output = resnet_model34(images)
                prediction = torch.max(output,1)[1]
                pred_test.append(prediction.cpu().numpy()[0])
                correct += (prediction.cpu().numpy()[0] == labels)
                test_labels_list.append(labels)
            print("Accuracy :",(correct/len(test_dataloader.dataset))*100,"%")
            
        confusion_matrix_test=confusion_matrix(pred_test, test_labels_list)
        plt.figure(figsize=(16,12))
        sns.heatmap(confusion_matrix_test, annot=True)
        plt.show()
    
    
    train_loader=train_data_loader(train_data,100)
    classes=train_loader.dataset.classes


    
    resnet_model34 = torchvision.models.resnet34(pretrained=True)

    for param in resnet_model34.parameters():
        param.requires_grad = False
    
    features_inp = resnet_model34.fc.in_features
    resnet_model34.fc = torch.nn.Linear(features_inp, 29)
    
    resnet_model34.to(device)
    #summary(resnet_model, (3, 200, 200), batch_size=100)
    learning_rates=[0.001] #0.01,0.001,0.00001
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    print("Start to train" )
    
    
    for l_rate in learning_rates:
        print(l_rate)
        loss_function_resnet=nn.CrossEntropyLoss()
        optimizer_resnet=torch.optim.Adam(resnet_model34.parameters(),
                                        lr=l_rate,weight_decay=0.00001)
        Model_train(10, train_loader,resnet_model34,
                    loss_function_resnet,optimizer_resnet)
        
        
def resNet18():
    train_transform=transforms.Compose([transforms.ColorJitter(brightness=0.3, 
                            saturation=0.1, contrast=0.1),transforms.ToTensor()])

    Train_data_path="./asl_alphabet_train/asl_alphabet_train/"
    train_data=ImageFolder(Train_data_path,transform=train_transform)
    print(len(train_data))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The device in use is {}".format(device)) 

    test_filepath = "./asl_alphabet_test/asl_alphabet_test/"
    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    test_dataset = torchvision.datasets.ImageFolder(test_filepath, transform=test_transforms)
    print("Test Dataset Info:\n",test_dataset)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=1)
    
 
    labels_map = {'A':0,'B':1,'C': 2, 'D': 3, 'E':4,'F':5,'G':6, 'H': 7, 'I':8, 'J':9,'K':10,'L':11, 'M': 12, 'N': 13, 'O':14, 
                    'P':15,'Q':16, 'R': 17, 'S': 18, 'T':19, 'U':20,'V':21, 'W': 22, 'X': 23, 'Y':24, 'Z':25, 
                    'del': 26, 'nothing': 27,'space':28}
    test_labels = []
    for folder_name in os.listdir(test_filepath):
        label = folder_name.replace("_test.jpg","")
        label = labels_map[label]
        test_labels.append(np.array(label))
    test_labels.sort()
        
    def train_data_loader(data_train, batchsize):
        train_loader=DataLoader(dataset=data_train, 
                                batch_size=batchsize,
                                shuffle=True, num_workers=2)
        return train_loader

    def Model_train(number_of_epochs, train_loader, Model, loss_function, optimizer):
        count=0
        training_accuracy = []
        training_lost = []
        for epoch in range(number_of_epochs):
            correct=0
            for images, labels in train_loader:
                count+=1
                images = images.cuda()
                labels = labels.cuda()
                outputs=Model(images)
                loss=loss_function(outputs, labels)
                # Back Propogation
                optimizer.zero_grad()
                loss.backward()
                # Update Weights (Optimize the model)
                optimizer.step()
                # Checking the performance
                predictions=torch.max(outputs,1)[1]
                correct+=(predictions==labels).cpu().sum().numpy()
                
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            training_accuracy.append(100*correct/len(train_loader.dataset))
            loss_string = str(loss.data)
            #training_lost.append(loss_string)
            print(current_time)
            print("Epoch is: {0}, Loss is {1} and Accuracy is: {2}".format(epoch+1,loss.data,100*correct/len(train_loader.dataset)))

        print("Training finished")
        
        plt.title("Training Accuracy Curve")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracu  (%)")
        plt.plot(range(1, number_of_epochs +1), np.array(training_accuracy), label = "Accuracy during the training" )
        plt.legend()
        plt.show()
        
        print("start to test")
    
        pred_test=[]
        test_labels_list=[]
        with torch.no_grad():
            correct = 0
            for (images,_),labels in zip(test_dataloader,test_labels):
                resnet_18.eval()
                images = images.to(device)
                output = resnet_18(images)
                prediction = torch.max(output,1)[1]
                pred_test.append(prediction.cpu().numpy()[0])
                correct += (prediction.cpu().numpy()[0] == labels)
                test_labels_list.append(labels)
            print("Accuracy :",(correct/len(test_dataloader.dataset))*100,"%")
            
        confusion_matrix_test=confusion_matrix(pred_test, test_labels_list)
        plt.figure(figsize=(16,12))
        sns.heatmap(confusion_matrix_test, annot=True)
        plt.show()
    
    
    train_loader=train_data_loader(train_data,100)
    classes=train_loader.dataset.classes


    resnet_18 = torchvision.models.resnet18(pretrained=True)
    for param in resnet_18.parameters():
        param.requires_grad = False
    
    features_inp = resnet_18.fc.in_features
    resnet_18.fc = torch.nn.Linear(features_inp, 29)
    
    resnet_18.to(device)
    #summary(resnet_model, (3, 200, 200), batch_size=100)
    learning_rates=[0.001] #0.01,0.001,0.00001
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    print("Start to train" )
    
    
    for l_rate in learning_rates:
        print(l_rate)
        loss_function_resnet=nn.CrossEntropyLoss()
        optimizer_resnet=torch.optim.Adam(resnet_18.parameters(),
                                        lr=l_rate,weight_decay=0.0001)
        Model_train(10, train_loader,resnet_18,
                    loss_function_resnet,optimizer_resnet)
        
        
if __name__ == '__main__':
    resNet34()
    #resNet18()