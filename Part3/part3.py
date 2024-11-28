import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram


# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_1.data", "rb")) # TODO remove this
old_dataset = pickle.load(open("../data2023/part3_dataset.data", "rb")) # TODO remove this


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
        model.fit(dataset)
        labels = model.labels_
        # labels = model.fit_predict(dataset)
        print(model.distances_)

        # silhouette analysis
        score = silhouette_score(dataset, labels)
        silhouette_scores.append(score)

        print(f" k={k}, silhouette score={score}")

        # keep track of the best k
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
            best_model = model
        
    # plot dendrogram
    plt.figure(figsize=(10, 7))
    plt.title(f"Dendrogram for HAC with linkage method: {linkage_method}, distance metric: {distance_metric}")
    plot_dendrogram(best_model)
    plt.show()

    plot_silhouette(k_vals, silhouette_scores, linkage_method, distance_metric)
    

    return best_score, best_k, best_labels, silhouette_scores

# plot silhoutte scores
def plot_silhouette(k_vals, silhouette_scores, linkage_method, distance_metric):
    plt.plot(k_vals, silhouette_scores, marker='o')
    plt.title(f"Silhouette Scores for HAC with linkage method: {linkage_method}, distance metric: {distance_metric}")
    plt.xlabel("K")
    plt.ylabel("Silhouette Score")
    plt.show()


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
            
            print(f"\nBest K for linkage method: {linkage_method}, distance metric: {distance_metric} is {k}")

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


test_hac(old_dataset)
