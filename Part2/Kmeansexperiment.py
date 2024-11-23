import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

# TODO remove old dataset
# dataset1 = pickle.load(open("../data2023/part2_dataset_1.data", "rb"))
# dataset2 = pickle.load(open("../data2023/part2_dataset_2.data", "rb"))

TIMES = 3 # TODO make it 10

def calc_confidence_interval(data):
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    h = 1.96 * std / np.sqrt(n)
    return (mean - h), (mean + h)

def get_min_kmeans_loss(dataset, k):
    min_loss = np.inf

    for _ in range(TIMES):
        km = KMeans(n_clusters=k)
        km.fit(dataset)
        min_loss = min(min_loss, km.inertia_)
    
    return min_loss

def calc_avg_kmeans_loss(dataset, k):
    losses = []

    for _ in range(TIMES):
        losses.append(get_min_kmeans_loss(dataset, k))

    return np.mean(losses), calc_confidence_interval(losses)

def get_avg_silhouette_score(dataset, k):
    sscores = [] 

    for _ in range(TIMES):
        km = KMeans(n_clusters=k)
        y_pred = km.fit_predict(dataset)
        sscores.append(silhouette_score(dataset, y_pred))

    return np.mean(sscores)

def calc_avg_silhouette_loss(dataset, k):
    scores = []

    for _ in range(TIMES):
        scores.append(get_avg_silhouette_score(dataset, k))
    
    return np.mean(scores), calc_confidence_interval(scores)

def calc_loss(dataset, k_range):
    avg_losses = []
    loss_conf_intervals = []

    for k in k_range:
        print(f"Running K means Loss for k = {k}")
        avg_loss, conf_int = calc_avg_kmeans_loss(dataset, k)
        avg_losses.append(avg_loss)
        loss_conf_intervals.append(conf_int)

        # report results
        # TODO better reporting to a csv file
        print(f"Average Loss = {round(avg_loss,3)}, Confidence Interval = {round(float(conf_int[0]),3), round(float(conf_int[1]),3)}")

    return avg_losses

def calc_silhouette_score(dataset, k_range):
    avg_scores = []
    score_conf_intervals = []

    for k in k_range:
        print(f"Running K means Silhouette score for k = {k}")
        avg_score, conf_int = calc_avg_silhouette_loss(dataset, k)
        avg_scores.append(avg_score)
        score_conf_intervals.append(conf_int)

        # report results
        # TODO better reporting to a csv file
        print(f"Average Silhouette Score = {round(avg_score,3)}, Confidence Interval = {round(float(conf_int[0]),3), round(float(conf_int[1]),3)}")

    return avg_scores

def save_plot(title, xlabel, ylabel , xs, ys):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(xs, ys)
    plt.grid()
    plt.savefig(f'plots/{title}') # TODO remove plots save location for grading
    plt.close()

# ploting
k_range = range(2,11)

print("Running K means loss method for Dataset 1")
ds1_yl = calc_loss(dataset1, k_range)
save_plot("K Means K vs Loss on Dataset 1", "Number of Clusters (K)", "Average Loss", k_range, ds1_yl)

print("Running K means loss method for Dataset 2")
ds2_yl = calc_loss(dataset2, k_range)
save_plot("K Means K vs Loss on Dataset 2", "Number of Clusters (K)", "Average Loss", k_range, ds2_yl)

print("Running K means Silhouette score method for Dataset 1")
ds1_ys = calc_silhouette_score(dataset1, k_range)
save_plot("K Means K vs Silhouette Score on Dataset 1", "Number of Clusters (K)", "Average Silhouette Score", k_range, ds1_ys)

print("Running K means Silhouette score method for Dataset 2")
ds2_ys = calc_silhouette_score(dataset2, k_range)
save_plot("K Means K vs Silhouette Score on Dataset 2", "Number of Clusters (K)", "Average Silhouette Score", k_range, ds2_ys)

