import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# The dataset is already preprocessed...
dataset = pickle.load(open("../datasets/part3_dataset.data", "rb"))

# plot Clustering Results
def plot_clusters(data, cluster_labels, method, name):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', s=1)
    plt.title(f"{method} Clustering on {name}")
    plt.colorbar()
    plt.savefig(f"dr_plots/{method}_{name}.png")
    plt.close()

# perform Clustering
def cluster_and_plot(reduced_data):
    for name, data in reduced_data.items():
        # dbscan 
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(data)
        sil_dbscan = silhouette_score(data, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
        plot_clusters(data, dbscan_labels, "DBSCAN", name)
        print(f"DBSCAN on {name} - Silhouette Score: {sil_dbscan:.2f}")
        
        # hac
        hac = AgglomerativeClustering(n_clusters=4, linkage='complete', metric='euclidean')
        hac_labels = hac.fit_predict(data)
        sil_hac = silhouette_score(data, hac_labels)
        plot_clusters(data, hac_labels, "HAC", name)
        print(f"HAC on {name} - Silhouette Score: {sil_hac:.2f}")
        
        # kmeans
        kmeans = KMeans(n_clusters=4)
        kmeans_labels = kmeans.fit_predict(data)
        sil_kmeans = silhouette_score(data, kmeans_labels)
        plot_clusters(data, kmeans_labels, "KMeans", name)
        print(f"KMeans on {name} - Silhouette Score: {sil_kmeans:.2f}")

# dimensionality reduction function
def reduce_dimensions(dataset):
    # PCA
    pca = PCA(n_components=2)
    pca_reduced = pca.fit_transform(dataset)
    
    # tsne
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    tsne_reduced = tsne.fit_transform(dataset)

    # tsne2 = TSNE(n_components=2, perplexity=50, learning_rate=200)
    # tsne_reduced2 = tsne2.fit_transform(dataset)
    
    # umap
    umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean')
    umap_reduced = umap.fit_transform(dataset)

    # umap2 = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, metric='euclidean')
    # umap_reduced2 = umap2.fit_transform(dataset)

    # umap3 = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine')
    # umap_reduced3 = umap3.fit_transform(dataset)

    # umap3 = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, metric='cosine')
    # umap_reduced5 = umap3.fit_transform(dataset)
    
    return {
        'PCA': pca_reduced,
        'TSNE perplexity=30': tsne_reduced,
        # 'TSNE perplexity=50': tsne_reduced2,
        'UMAP n_neighbors=15 euclidean': umap_reduced,
        # 'UMAP n_neighbors=5 euclidean': umap_reduced2,
        # 'UMAP n_neighbors=15 cosine': umap_reduced,
        # 'UMAP n_neighbors=5 cosine': umap_reduced2
    }

# reduce dimensions
reduced_data = reduce_dimensions(dataset)
cluster_and_plot(reduced_data)

