#from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np
import argparse


print("Celebrity Facial Classification")
face_data = fetch_lfw_people(min_faces_per_person=100, resize = 0.4) #fetches face data, only keeps people with at least 70 pictures
num_images = face_data.images.shape[0]
print(f'Num Images: {num_images}\nImage Shape: {face_data.images.shape[1]} by {face_data.images.shape[2]}')
print(f'Featured Celebrities: {face_data.target_names}')

trainX, testX, trainY, testY = train_test_split(
    face_data.data, face_data.target, test_size = 0.3, shuffle = False)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", required = True, help = "string with which classification model to use: LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, MLPClassifier, or SGDClassifier")
args = vars(parser.parse_args())
model = args["model"]

if model == "LogisticRegression":
    classifier = LogisticRegression('l2', max_iter=100, solver = 'liblinear', shuffle = False)

elif model == "LogisticRegressionCV":
    classifier = LogisticRegressionCV(penalty = 'l2', max_iter=100, solver = 'liblinear', shuffle = False)

elif model == "Perceptron":
    classifier = Perceptron(penalty = 'l1', alpha = 0.14, max_iter=100, shuffle = False)

elif model == "PassiveAggressiveClassifier":
    classifier = PassiveAggressiveClassifier(max_iter=100, shuffle = False)

elif model == "SGDClassifier":
    classifier = SGDClassifier(alpha = 0.12, eta0 = 0.04, learning_rate = "adaptive", tol = 0.001,  max_iter=1000, shuffle = False)

else:
    print("Error: Input one of the following models listed in help")


classifier.fit(trainX, trainY)
preds = classifier.predict(testX)
print(preds.shape)


accuracy = accuracy_score(testY, preds)
print(f'Accuracy: {accuracy}')
plot_confusion_matrix(classifier, testX, testY)
plt.show()