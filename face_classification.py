#from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, plot_confusion_matrix, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np

print("Celebrity Facial Classification")
face_data = fetch_lfw_people(min_faces_per_person=100, resize = 0.4) #fetches face data, only keeps people with at least 70 pictures
num_images = face_data.images.shape[0]
print(f'Num Images: {num_images}\nImage Shape: {face_data.images.shape[1]} by {face_data.images.shape[2]}')
print(f'Featured Celebrities: {face_data.target_names}')

trainX, testX, trainY, testY = train_test_split(
    face_data.data, face_data.target, test_size = 0.3, shuffle = True)

classifier = LogisticRegression('l2', max_iter=2000) #later fine-tune for better results
classifier.fit(trainX, trainY)
preds = classifier.predict(testX)
print(preds.shape)

accuracy = accuracy_score(testY, preds)
print(accuracy)
plot_confusion_matrix(classifier, testX, testY)
plt.show()