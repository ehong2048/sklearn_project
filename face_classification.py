from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, mean_squared_error, r2_score
from matplotlib import pyplot as plt
import numpy as np

print("Celebrity Facial Classification")
lfw_people = fetch_lfw_people(min_faces_per_person=60) #fetches face data, only keeps people with at least 70 pictures
num_images = {lfw_people.images.shape[0]}
print(f'Num Images: {num_images}\nImage Size: {lfw_people.images.shape[1]} x {lfw_people.images.shape[2]}')

