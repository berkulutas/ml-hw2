import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram

from sklearn.cluster import DBSCAN

# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_1.data", "rb")) # TODO remove this
old_dataset = pickle.load(open("../data2023/part3_dataset.data", "rb")) # TODO remove this

##### HAC #####

# hac clustering and silhouette analysis
def hac_and_silhouette(dataset, linkage_method, distance_metric, k_vals):
    best_score = -1
    best_k = None
    best_labels = None
    best_model = None
    silhouette_scores = []

    print(f"\nTesting HAC with linkage method: {linkage_method}, distance metric: {distance_metric}")

    for k in k_vals:
        # hac clustering
        # fix labels_ https://stackoverflow.com/questions/61362625/agglomerativeclustering-no-attribute-called-distances
        model = AgglomerativeClustering(linkage=linkage_method, metric=distance_metric, n_clusters=k, compute_distances=True) # compute_distances for fix labels_
        labels = model.fit_predict(dataset)

        # silhouette analysis
        score = silhouette_score(dataset, labels)
        silhouette_scores.append(score)

        print(f" k={k}, silhouette score={score:.4f}")

        # keep track of the best k
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = model
        
    # plot dendrogram
    plt.figure(figsize=(10, 7))
    title = f"Dendrogram for HAC (Linkage Method: {linkage_method}, Distance Metric: {distance_metric}, K: {best_k})"
    plt.title(title)
    plot_dendrogram(best_model, truncate_mode='level', p=best_k)
    plt.savefig(f'plots/{title}.png')
    plt.close()

    plot_hac_silhouette(k_vals, silhouette_scores, linkage_method, distance_metric)
    

    return best_score, best_k, best_labels, silhouette_scores

# plot silhoutte scores
def plot_hac_silhouette(k_vals, silhouette_scores, linkage_method, distance_metric):
    plt.rcParams["figure.figsize"] = (10, 7)
    plt.plot(k_vals, silhouette_scores, marker='o')
    title = f"K vs Silhouette Scores for HAC (Linkage Method: {linkage_method}, Distance Metric: {distance_metric})"
    plt.title(title)
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(f'plots/{title}.png')
    plt.close()


# run hac clustering and silhouette analysis
def test_hac(dataset):
    best_overall_score = -1
    best_overall_conf = None
    k_vals = range(2,6)

    for linkage_method in ['single', 'complete']:
        for distance_metric in ['euclidean', 'cosine']:
            score, k, labels, silhouette_scores = hac_and_silhouette(dataset, linkage_method, distance_metric, k_vals)

            if score > best_overall_score:
                best_overall_score = score
                best_overall_conf = (linkage_method, distance_metric, k, labels, silhouette_scores)
            
            print(f"\nBest K:{k}; for linkage method: {linkage_method}, distance metric: {distance_metric}")

    print(f"\nBest overall configuration: {best_overall_conf[0]}, {best_overall_conf[1]}, {best_overall_conf[2]}")

    return best_overall_conf

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


##### DBSCAN #####

# dbscan clustering and silhouette analysis
def dbscan_and_silhouette(dataset, metric_and_eps, min_samples_vals):
    results = []

    print("\nTesting DBSCAN with multiple configurations...")
    for metric, eps_vals in metric_and_eps:
        for eps in eps_vals:
            for min_samples in min_samples_vals:
                print(f"testing DBSCAN (eps={eps:.2f}, min_samples={min_samples}, metric={metric}")
                # apply dbscan
                model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
                labels = model.fit_predict(dataset) # -1 for noise

                # skip no cluster or noise
                uniqe_labels = set(labels)
                # print(uniqe_labels)
                if (len(uniqe_labels) == 1):# or (-1 in uniqe_labels and len(uniqe_labels) == 2):
                    continue

                # silhouette analysis
                score = silhouette_score(dataset, labels) 
                results.append((eps, min_samples, metric, score, labels, len(uniqe_labels)))
                print(f"DBSCAN (eps={eps:.2f}, min_samples={min_samples}, metric={metric}, silhouette score={score:.4f}), unique labels count: {len(uniqe_labels)}")

    return results

def plot_eps_vs_silhouette(results):
    # y axis is silhouette score 
    # x axis is eps
    # color is min_samples
    # different plots for different metrics
    plt.rcParams["figure.figsize"] = (10, 7)
    for metric in ['euclidean', 'cosine']:
        plt.figure()
        for min_samples in [2, 3, 4, 5]:
            eps_vals = [result[0] for result in results if result[2] == metric and result[1] == min_samples]
            silhouette_scores = [result[3] for result in results if result[2] == metric and result[1] == min_samples]
            plt.plot(eps_vals, silhouette_scores, marker='o', label=f"min_samples={min_samples}")
        plt.title(f"DBSCAN Silhouette Scores vs Eps for {metric}")
        plt.xlabel("Eps")
        plt.ylabel("Silhouette Score")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'plots_dbscan/DBSCAN Silhouette Scores vs Eps for {metric}.png')
        plt.close()


# # plot silhoutte scores
# def plot_dbscan_silhouette(dataset, labels, eps, min_samples, metric, rank):
#     plt.rcParams["figure.figsize"] = (10, 7)
#     silhouette_vals = silhouette_samples(dataset, labels)
#     y_lower = 10
#     for i in range(len(set(labels))):
#         ith_cluster_silhouette_vals = silhouette_vals[labels == i]
#         ith_cluster_silhouette_vals.sort()
#         size_cluster_i = ith_cluster_silhouette_vals.shape[0]
#         y_upper = y_lower + size_cluster_i
#         color = plt.cm.nipy_spectral(float(i) / len(set(labels)))
#         plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_vals, facecolor=color, edgecolor=color, alpha=0.7)
#         plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
#         y_lower = y_upper + 10

#     title = f"Silhouette Analysis for DBSCAN (eps={eps:.2f}, min_samples={min_samples}, metric={metric}, Rank {rank})"
#     plt.title(title)
#     plt.xlabel("Silhouette Coefficient Values")
#     plt.ylabel("Cluster label")
#     plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--")
#     plt.savefig(f'plots_dbscan/{title}.png')
#     plt.close()

# run dbscan clustering and silhouette analysis
def test_dbscan(dataset):
    # hyperparameters
    # eps_vals = [14, 15, 16, 17, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    min_samples_vals = [2, 5, 10, 15, 30]
    # distance_metrics = ['cosine', 'euclidean']
    metric_and_eps = [('euclidean', [13, 14, 15, 16, 17, 18, 19, 20]), ('cosine', [0.1, 0.12, 0.14, 0.16, 0.18, 0.2])]

        
    results = dbscan_and_silhouette(dataset, metric_and_eps, min_samples_vals)
    
    plot_eps_vs_silhouette(results)

    # sort results by silhouette score
    results.sort(key=lambda x: x[3], reverse=True)
    best_configs = enumerate(results[:4], start=1)

    print("\nBest 4 configurations for DBSCAN:")
    for rank, config in best_configs:
        print(f"Rank {rank}: eps={config[0]:.2f}, min_samples={config[1]}, metric={config[2]}, silhouette score={config[3]:.4f}, K={config[5]}")
        # plot_dbscan_silhouette(dataset, config[4], config[0], config[1], config[2], rank)


test_hac(dataset)
test_dbscan(dataset) 