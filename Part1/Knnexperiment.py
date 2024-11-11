import pickle
from Distance import Distance
from Part1.Knn import KNN

# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))
