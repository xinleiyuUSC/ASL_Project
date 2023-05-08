# ASL_Project
This is the Github for EE541 final project - American Sign Language.
Author: Xinlei Yu and Xingjue Liao

Deep Learning for American Sign Language (ASL) recognition has gained significant attention in recent years, with researchers using various datasets available on Kaggle to develop innovative models. The dataset linked above contains images from 29 classes (26 alphabets, SPACE, DELETE and NOTHING). Each class contains 3000 images in the training set and each image is a 200 x 200 RGB image This dataset is instrumental in training deep learning models to automatically recognize and translate ASL gestures into text or speech, thus improving communication for the deaf and hard-of-hearing community.

In deep learning, convolutional Neural Networks (CNNs) have emerged as the leading approach in developing such models due to their ability to efficiently process and analyze visual data.

/Datasets
/CNN
/ResNet

Instruction of running ResNet_ASL.py
It's pretty straightforward to run the ResNet_ASL.py. Firstly, make sure we have installed PyTorch, NumPy, Seaborn, and etc. as they are imported in the python scripts.
For the test sets, makesure you also include each class of image into a folder, such that "test_dataset/A/A_image". 
'''shell
conda activate "your project name"

python ResNet_ASL.py

'''
