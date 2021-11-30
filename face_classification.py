from sklearn.datasets import fetch_lfw_people
from sklearn.externals._pilutil import imread
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report, balanced_accuracy_score
from matplotlib import pyplot as plt
import argparse
import numpy as np
import cv2

print("\n---Politician Facial Classification---")

# parse arguments for which classification model and for string path to what face image to classify
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model", required = True, help = "string with which classification model to use: LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, or SGDClassifier")
parser.add_argument('-i', "--image", help = "str path to image of Powell, Bush, or Blair to classify")
args = vars(parser.parse_args())
model = args["model"]

# loads face data and prints information
face_data = fetch_lfw_people(data_home="./test-dl", min_faces_per_person=140, color = True) # fetches face data, only keeps people with at least 140 pictures (Bush, Blair, Powell)
num_images = face_data.images.shape[0]
h = face_data.images.shape[1]
w = face_data.images.shape[2]
print(f'Num Images: {num_images}\nImage Shape: {h} by {w}')
labels = face_data.target_names # gets the labels of the data
print(f'Featured Politicians: {labels}')

# splits data into test set and training set
trainX, testX, trainY, testY = train_test_split(
    face_data.data, face_data.target, test_size = 0.3, shuffle = False)

# creates classification model (each one is fine-tuned)
if model == "LogisticRegression":
    classifier = LogisticRegression('l2', max_iter=100, solver = 'liblinear')

elif model == "LogisticRegressionCV":
    classifier = LogisticRegressionCV(penalty = 'l2', max_iter=100, solver = 'liblinear')

elif model == "Perceptron":
    classifier = Perceptron(penalty = 'l1', alpha = 0.14, max_iter=100, shuffle = False)

elif model == "PassiveAggressiveClassifier":
    classifier = PassiveAggressiveClassifier(max_iter=100, shuffle = False)

elif model == "SGDClassifier":
    classifier = SGDClassifier(alpha = 0.12, eta0 = 0.04, learning_rate = "adaptive", tol = 0.001,  max_iter=1000, shuffle = False)

else:
    print("Error: Input one of the following models listed in help")
    quit()

# fits and trains model, then gets prediction for test set
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)

# prints metrics for test set and displays confusion matrix
print("\n---Classification Model Metrics---")
accuracy = accuracy_score(testY, preds)
print(f'Accuracy: {accuracy}')
balanced_accuracy = balanced_accuracy_score(testY, preds)
print(f'Balanced Accuracy (Average Recall): {balanced_accuracy}')
report = classification_report(testY, preds, target_names=labels)
print(f'\nClassification Report:\n{report}')
plot_confusion_matrix(classifier, testX, testY)
plt.show()


print("\n---Sanity Check: Testing Classifier on Image from actual Kaggle Dataset---")
test = face_data.data[10].reshape(1, -1)
face_pred = classifier.predict(test)
print(f'Target: {face_data.target_names[face_data.target[10]]}')
print(f'Predicted: {face_data.target_names[face_pred]}')
cv2.imshow("test", (test/255.0).reshape(62, 47, 3)) # have to divide by 255 so that we can see image (caused by how data is processed in fetch function)
cv2.waitKey(0)



print("\n---Sanity Check 2: Testing Classifier on Image from Google---")
# loads face image based on passed in str path
face_path = args["image"]
img = imread(face_path) # uses sklearn's imread function (to be consistent with sklearn's _lfw_people function), which processes it differently 
cv2.imshow(f'Face to classify', img)
cv2.waitKey()

# transforms image to be consistent with Kaggle dataset
face = np.asarray(img, dtype = np.float32)
print(f'Face shape: {face.shape}')
face /= 255.0 # dividies by 255 (which is how it's processed in sklearn's fetch_lfw_people function)
dim = (face_data.images.shape[2], face_data.images.shape[1])
face = cv2.resize(face, dim, interpolation = cv2.INTER_AREA) # resizes to same size as Kaggle's face data
face = np.array(face).flatten().reshape(1, -1) # flattens to be 1-D like Kaggle data
print(f'New face shape: {face.shape}')

# uses transformed image data to predict its politician identity
face_pred = classifier.predict(face)
print(f'Predicted: {face_data.target_names[face_pred]}')

