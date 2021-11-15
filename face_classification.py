from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, plot_confusion_matrix
from matplotlib import pyplot as plt
import argparse
import numpy as np
import cv2

print("---Celebrity Facial Classification---")

parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model", required = True, help = "string with which classification model to use: LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, or SGDClassifier")
parser.add_argument('-i', "--image", help = "str path to image of Powell, Rumsfeld, Bush, or Schroeder to classify")
args = vars(parser.parse_args())
model = args["model"]

face_data = fetch_lfw_people(min_faces_per_person=140, resize = 0.4, color = True) #fetches face data, only keeps people with at least 70 pictures
print(face_data.data.shape)
num_images = face_data.images.shape[0]
h = face_data.images.shape[1]
w = face_data.images.shape[2]
print(f'Num Images: {num_images}\nImage Shape: {h} by {w}')
print(f'Featured Celebrities: {face_data.target_names}')

trainX, testX, trainY, testY = train_test_split(
    face_data.data, face_data.target, test_size = 0.3, shuffle = False)

"""
n_components = 150
pca = PCA(n_components=n_components, whiten=True).fit(trainX)
eigenfaces = pca.components_.reshape((n_components, h, w))

trainX = pca.transform(trainX)
testX = pca.transform(testX)
"""

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

classifier.fit(trainX, trainY)
preds = classifier.predict(testX)
print(preds.shape)

accuracy = accuracy_score(testY, preds)
print(f'Accuracy: {accuracy}')
plot_confusion_matrix(classifier, testX, testY)
plt.show()



face_path = args["image"]
face = cv2.imread(face_path)
dim = (face_data.images.shape[2], face_data.images.shape[1])
# face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
face = cv2.resize(face, dim, interpolation = cv2.INTER_AREA)
cv2.imshow(f'Face to classify', face)
cv2.waitKey(0)

face = np.ndarray.flatten(face)
face = np.expand_dims(face, 0)
print(face.shape)
face_pred = classifier.predict(face)
print(face_pred)
print(face_data.target_names[face_pred])

