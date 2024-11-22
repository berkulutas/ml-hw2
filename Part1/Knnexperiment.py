import pickle
from Distance import Distance
from Knn import KNN

from sklearn.model_selection import StratifiedKFold # TODO questoin? can we import this

import random # TODO question? can we import this

# the data is already preprocessed
dataset, labels = pickle.load(open("../datasets/part1_dataset.data", "rb"))

iterations = 5
fold_num = 10
hyperparameter_configs = {
    "k": [3, 5, 10, 30, 50], # Typically, K values of 5, 10, or 30 are considered.
    "distance_fn": [Distance.calculateCosineDistance, Distance.calculateMinkowskiDistance, Distance.calculateMahalanobisDistance] # TODO minkowski with different p parameters
}
train_test_split = 0.8

def calculate_conf_interval_mean(data):
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5
    std_error = std / (len(data) ** 0.5)
    margin_of_error = 1.96 * std_error
    return round(mean - margin_of_error, 3), round(mean + margin_of_error,3 ), round(mean,3)

# shuffle the data
indices = list(range(len(dataset)))
random.seed(31)
random.shuffle(indices)
dataset = dataset[indices]
labels = labels[indices]

# split train and test data
split_index = int(len(dataset)*train_test_split)
train_data, train_labels = (dataset[:split_index], labels[:split_index])
test_data, test_labels =   (dataset[split_index:], labels[split_index:])

best_hyperparameters = None
best_accuracy = 0

for k in hyperparameter_configs["k"]:
    for distance_fn in hyperparameter_configs["distance_fn"]:
        accuracies = []
        curr_config = {"k": k, "distance_fn": distance_fn}
        # iterate 5 times for statistical significance
        for _ in range(iterations):
            # stratified k-fold
            skf = StratifiedKFold(n_splits=fold_num, shuffle=True)
            for train, test in skf.split(train_data, train_labels):
                knn = KNN(train_data[train], train_labels[train], distance_fn, K=k)
                correct = 0
                for i in range(len(test)):
                    if knn.predict(train_data[test[i]]) == train_labels[test[i]]:
                        correct += 1
                accuracies.append(correct/len(test))
        # calculate mean and confidence interval
        lower, upper, mean = calculate_conf_interval_mean(accuracies)
        # TODO better reporting
        print(f"K: {k}, Distance Function: {distance_fn.__name__}, Accuracy: {mean}, Confidence Interval: [{lower}, {upper}]")
        if mean > best_accuracy:
            best_accuracy = mean
            best_hyperparameters = curr_config
# TODO better reporting
print(f'Best Hyperparameters:\nDistance Function: {best_hyperparameters["distance_fn"].__name__}, K: {best_hyperparameters["k"]}\nBest Accuracy: {best_accuracy}')

# test the best hyperparameters on the test data
knn = KNN(train_data, train_labels, best_hyperparameters["distance_fn"], K=best_hyperparameters["k"])
correct = 0
for i in range(len(test_data)):
    if knn.predict(test_data[i]) == test_labels[i]:
        correct += 1
print(f"Test Accuracy: {correct/len(test_data)}")
        # TODO better reporting