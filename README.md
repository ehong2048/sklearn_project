# sklearn_project
Author: Emma Hong
11/2021 Open Scikit Learn Project to demonstrate understanding

## Description
I use skikit-learn's Labeled Faces in the Wild dataset to train a classification model for classifying images of Bush, Powell, and Blair, acheiving an accuracy of 90%. Then, I use it to classify downloaded images of the 3 people from Google images to see if it works.

## Usage
    python face_classification.py --model $model_name$ --image $path_image_to_classify$

model_name are any of the following strings: LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, or SGDClassifier

## Classification Metric Explanations
(Each model was fine-tuned to optimize the accuracy. I also made Shuffle = False so that the accuracy stayed the same for different trials while fine-tuning.)

Accuracy: fraction of correct predictions over number of samples
Balanced_Accuracy: Macro average of recall (accounts for imbalanced datasets such as this one, where Bush images dominate)

Precision: ability of the classifier not to label as positive a sample that is negative (best is 1)
> P = tp / (tp + fp) where tp is true positives and fp is false positives

Recall: ability of the classifier to find all positive samples (best is 1)
> R = tp / (tp + fn) where tp is true positives and fn is false negatives

F1-score: harmonic mean of precision and recall (best is 1)
> F1 = 2 * (precision * recall) / (precision + recall)

Support: Number of true instances of each class

Macro Average: unweighted mean of metric for the different classes

Weighted Average: mean of metric for the different classes weighted by each class' support (accounts for label imbalance)

## Each Model's Metrics and Analysis
LogisticRegression: 

&nbsp;&nbsp;Accuracy = 89.4%, Balanced_Accuracy = 86.8%

LogisticRegressionCV:
* Accuracy = 88.3%, Balanced_Accuracy = 85.5%
* *(Takes a significantly longer time to train model)

PassiveAggressiveClassifier:
* Accuracy = 88.3%, Balanced_Accuracy = 81.7%
* (0.97 Bush Recall but 0.86 precision due to lots of false positives for Bush, i.e. predicted Bush a lot. Also caused low recall for Powell and Blair as a result of falsely labelling them as Bush.)

Perceptron:
* Accuracy = 81.7%, Balanced_Accuracy = 70.6%
* (0.98 Bush Recall but 0.77 precision due to lots of false positives for Bush, i.e. predicted Bush a lot. Also caused low recall for Powell and Blair as a result of falsely labelling them as Bush.)

**SGDClassifier:**
* Accuracy = 90.1%, Balanced_Accuracy = 86.7%
* (Slightly faster than LogisticRegression)

For all of the models, Bush had consistently higher metrics (precision, recall, f1-score) for the most part, which makes sense considering that he had 530 total images in the dataset compared to 236 for Powell and 144 for Blair. This reflects the biases inherent in data. This translates into the real world negatively by ingraining prejudices into the system (ex. IBM Facial Recognition failing to identify darker-skinned women).

## SGD Classifier Detailed Metrics
Accuracy (Average Precision): 0.9010989010989011
Balanced Accuracy (Average Recall): 0.8672249269717623
Classification Report:

              |  Prec. | Recall |   F1   |  Supp.
------------- | ----------------------------------
Colin Powell  |  0.87  |  0.89  |  0.88  |  65
George W Bush |  0.91  |  0.95  |  0.93  |  158
Tony Blair    |  0.90  |  0.76  |  0.83  |  50
              |        |        |        |    
Accuracy      |        |        |  0.90  |  273
Macro Avg.    |  0.90  |  0.87  |  0.88  |  273
Weighted Avg. |  0.90  |  0.90  |  0.90  |  273