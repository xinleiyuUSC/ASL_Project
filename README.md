# ASL_Project
This is the Github for EE541 final project - American Sign Language.
Author: Xinlei Yu and Xingjue Liao

## Introduction
Deep Learning for American Sign Language (ASL) recognition has gained significant attention in recent years, with researchers using various datasets available on Kaggle to develop innovative models. The dataset linked above contains images from 29 classes (26 alphabets, SPACE, DELETE and NOTHING). Each class contains 3000 images in the training set and each image is a 200 x 200 RGB image This dataset is instrumental in training deep learning models to automatically recognize and translate ASL gestures into text or speech, thus improving communication for the deaf and hard-of-hearing community.

In deep learning, convolutional Neural Networks (CNNs) have emerged as the leading approach in developing such models due to their ability to efficiently process and analyze visual data.

## Contents
There are a few files and folder inside of this github repo. Although, dataset is not required to list at here, we put our additional dataset here for reference. 

- Input folder contains our self created data sets to test the deep learning models. Test set 1 and 2 are two new test sets generated from Internet images.

- Test set 3 is the high contrasted and shaped images sets made from test set 2.

- Output folder contains part of plot diagrams and confusion matrixs. Some CNN models are also stored here, described by the Readme within the folder.

- ASL-CNN-test.py is a python script that loads a existing CNN model to test it. 

- ASL-CNN.py is a python script contains CNN architecture and also the training code. 

- ResNet_ASL.py is a python script contains ResNet architecture deep learning mode with ResNet18 and ResNet34. 
## Instruction of running ASL-CNN.py
- Make sure you have all dependencies installed (PyTorch, Torchvision,...)

- Open ASL-CNN.py in IDE (I used Spyder)

- Change the following to the approriate file path of training set in your folder
```
Train_data_path="ASL/asl_alphabet_train/asl_alphabet_train"
```
- Change the following to the approriate file path of output location in your folder
```
plt.savefig('outputs/accuracy.png')
plt.savefig('outputs/loss.png')
torch.save(Model_1.state_dict(), 'outputs/model.pth')
```
- Run the code

## Instruction of running ASL-CNN-test.py
- Make sure you have all dependencies installed (PyTorch, Torchvision,...)

- Open ASL-CNN-test.py in IDE (I used Spyder)

- Change the following to the approriate file path of test set in your folder
```
test_filepath = "ASL/asl_alphabet_test/"
test_filepath = "ASL/asl_alphabet_test/asl_alphabet_test/"
```
- Change the following to the approriate file extension of the images in your test set
```
label = folder_name.replace("_test.jpg","")
```
- Change the following to the approriate  file path of output files in your folder
```
Model_1.load_state_dict(torch.load('outputs/model.pth'))
```
- Run the code

## Instruction of running ResNet_ASL.py
It's pretty straightforward to run the ResNet_ASL.py. Firstly, make sure we have installed PyTorch, NumPy, Seaborn, and etc. as they are imported in the python scripts.
For the test sets, makesure you also include each class of image into a folder, such that "test_dataset/A/A_image". The defualt epochs is set to 10 and it will run ResNet34 model. To switch to Resnet16, simply uncomment the line of code, then it should work. 

```shell
conda activate "your project name"

python ResNet_ASL.py

```
