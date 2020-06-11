# American Sign Language Recognition using Deep Learning
## 1 - Introduction
American Sign Language (ASL) is a complete, natural language that is expressed using the movement of hands and face. ASL provides the deaf community a way to interact within the community itself as well as to the outside world.
With the advent of [Artificial Neural Networks](https://medium.com/technology-invention-and-more/everything-you-need-to-know-about-artificial-neural-networks-57fac18245a1) and [Deep Learning](https://www.mathworks.com/discovery/deep-learning.html), it is now possible to build a system that can recognize objects or even objects of various categories (like red vs green apple). Utilizing this, here we have an application that uses a deep learning model trained on the ASL Dataset to predict the sign from the sign language given an input image or frame from a video feed.
You can learn more about the American Sign Language over [here](https://www.nidcd.nih.gov/health/american-sign-language) [National Institute on Deafness and Other Communication Disorders (NIDCD) website].

#### Alphabet signs in American Sign Language are shown below:
![American Sign Language - Signs](/images/NIDCD-ASL-hands-2014.jpg)

## 2 - Approach
We will utilize a method called [Transfer Learning](https://machinelearningmastery.com/transfer-learning-for-deep-learning/) along with [Data Augmentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b) to create a deep learning model for the ASL dataset.

### 2.1 - Dataset
The network was trained on this kaggle dataset of [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet). The dataset contains `87,000` images which are 200x200 pixels, divided into `29` classes (`26` English Alphabets and `3` additional signs of SPACE, DELETE and NOTHING). 

### 2.2 - Transfer Learning (Inception v3 as base model)
The network uses Google's [Inception v3](https://arxiv.org/pdf/1512.00567.pdf) as the base model.

### 2.3 - Using the model for the application
After the model is trained, it is then loaded in the application. [OpenCV](https://opencv.org/) is used to capture frames from a video feed. The application provides an area (inside the green rectangle) where the signs are to be presented to be detected or recognized. 
Confidence above 50% is represented as `HIGH`.
Between `20%` to `50%` is `LOW`
Else the model displays `Nothing`

## 3 - Results
For training, [Categorical Crossentropy](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy) was used to measure the loss along with [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) optimizer (with `learning rate` of `0.0001` and `momentum` of `0.9`) to optimize our model. The model is trained for `24` epochs. The results are displayed below:

### 3.1 - Tabular Results
Metric | Value
-------|------
Training Accuracy | 0.9356 (~93.56%)
Training Loss | 0.1700
Validation Accuracy | 0.9275 (~92.75%)
Validation Loss | 0.1626
Test Accuracy | 91.43%


## 4 - Running the application
If you want to try out the application, you might have to satisfy some requirements to be able to run it on your PC.

### 4.1 - Requirements
- [Python](https://www.python.org/downloads/) v3.7.4 or higher (should work with v3.5.2 and above as well)
- [NumPy](https://www.scipy.org/install.html)
- [OpenCV](https://solarianprogrammer.com/2016/09/17/install-opencv-3-with-python-3-on-windows/) v3 or higher
- [Tensorflow](https://www.tensorflow.org/install) v1.15.0 (may work on higher versions)[GPU version preferred]
- Might require a PC with NVIDIA GPU (at least 4GB graphics memory)
- Webcam

### 4.2 - Clone this repository
- Clone this repository using 

### 4.3 - Executing the script
1. Open a command prompt inside the cloned repository folder or just open a command prompt and navigate to the cloned directory.
1. Execute this command: `python asl_alphabet_application.py`.
1. An application opens up after a few minutes.

