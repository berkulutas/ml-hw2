import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids

from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

km = KMedoids(n_clusters=2)

# TIMES = 3 # TODO make it 10

# def calc_confidence_interval(data):
#     mean = np.mean(data)
#     std = np.std(data)
#     n = len(data)
#     h = 1.96 * std / np.sqrt(n)
#     return (mean - h), (mean + h)

# def get_min_kmedoids_loss(dataset, k):
#     min_loss = np.inf

#     for _ in range(TIMES):
#         km = KMedoids(n_clusters=k)
#         km.fit(dataset)
#         min_loss = min(min_loss, km.inertia_)
    
#     return min_loss

# def calc_avg_kmedoids_loss(dataset, k):
#     losses = []

#     for _ in range(TIMES):
#         losses.append(get_min_kmedoids_loss(dataset, k))

#     return np.mean(losses), calc_confidence_interval(losses)

# def get_min_silhouette_score(dataset, k):
#     min_ss = np.inf

#     for _ in range(TIMES):
#         km = KMedoids(n_clusters=k)
#         y_pred = km.fit_predict(dataset)
#         min_ss = min(min_ss, silhouette_score(dataset, y_pred))
    
#     return min_ss

# def calc_avg_silhouette_loss(dataset, k):
#     scores = []

#     for _ in range(TIMES):
#         scores.append(get_min_silhouette_score(dataset, k))
    
#     return np.mean(scores), calc_confidence_interval(scores)

# def calc_loss(dataset, k_range):
#     avg_losses = []
#     loss_conf_intervals = []
#     avg_silhouette_scores = []
#     silhouette_conf_intervals = []

#     for k in k_range:
#         avg_loss, loss_conf_interval = calc_avg_kmedoids_loss(dataset, k)
#         avg_losses.append(avg_loss)
#         loss_conf_intervals.append(loss_conf_interval)

#         avg_silhouette_score, silhouette_conf_interval = calc_avg_silhouette_loss(dataset, k)
#         avg_silhouette_scores.append(avg_silhouette_score)
#         silhouette_conf_intervals.append(silhouette_conf_interval)
    
#     return avg_losses, loss_conf_intervals, avg_silhouette_scores, silhouette_conf_intervals

# k_range = range(2, 11)
# avg_losses1, loss_conf_intervals1, avg_silhouette_scores1, silhouette_conf_intervals1 = calc_loss(dataset1, k_range)
# avg_losses2, loss_conf_intervals2, avg_silhouette_scores2, silhouette_conf_intervals2 = calc_loss(dataset2, k_range)

# plt.figure()
# plt.errorbar(k_range, avg_losses1, yerr=np.array(loss_conf_intervals1).T)
# plt.errorbar(k_range, avg_losses2, yerr=np.array(loss_conf_intervals2).T)
# plt.xlabel("Number of clusters")
# plt.ylabel("Loss")
# plt.title("KMedoids Loss")
# plt.legend(["Dataset 1", "Dataset 2"])
# plt.savefig("KMedoids_loss.png")

# plt.figure()
# plt.errorbar(k_range, avg_silhouette_scores1, yerr=np.array(silhouette_conf_intervals1).T)
# plt.errorbar(k_range, avg_silhouette_scores2, yerr=np.array(silhouette_conf_intervals2).T)
# plt.xlabel("Number of clusters")
# plt.ylabel("Silhouette Score")
# plt.title("KMedoids Silhouette Score")
# plt.legend(["Dataset 1", "Dataset 2"])
# plt.savefig("KMedoids_silhouette_score.png")
# # save_plot("KMedoids Loss", "Number of clusters", "Loss", k_range, avg_losses1)
# # save_plot("KMedoids Loss", "Number of clusters", "Loss", k_range, avg_losses2)
# # save_plot("KMedoids Silhouette Score", "Number of clusters", "Silhouette Score", k_range, avg_silhouette_scores1)
# # save_plot("KMedoids Silhouette Score", "Number of clusters", "Silhouette Score", k_range, avg_silhouette_scores2)

