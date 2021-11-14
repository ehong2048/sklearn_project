# sklearn_project
Author: Emma Hong
11/2021 Open Scikit Learn Project to demonstrate understanding

## Description
I use skikit-learn's Labeled Faces in the Wild dataset to train a classification model for classifying images of Bush, Powell, Rumsfeld, Schroeder, and Blair, acheiving an accuracy of 86%. Then, I use it to classify downloaded images of the 5 people from the internet to see if it works.

## Usage
    python face_classification.py --model $model_name$ --image $path_image_to_classify$

model_name are any of the following strings: LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, or SGDClassifier