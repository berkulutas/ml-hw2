from sklearn.neighbors import KNeighborsClassifier
import sklearn.datasets
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score

dataset, label = sklearn.datasets.load_wine(return_X_y=True)
print(dataset.shape)

kfold = RepeatedStratifiedKFold()

for train_indices, test_indices in kfold.split(dataset, label):
    current_train = dataset[train_indices]
    current_train_label = label[train_indices]

    knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski', p=3)
    knn.fit(current_train, current_train_label)

    curret_test = dataset[test_indices]
    current_test_label =  label[test_indices]

    prediced = knn.predict(curret_test)
    accuracy = accuracy_score(current_test_label, prediced)
    print(f"accuracy = {round(accuracy, 3)}")