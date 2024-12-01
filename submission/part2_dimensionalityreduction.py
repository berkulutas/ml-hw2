import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pca import PCA
from autoencoder import AutoEncoder
from sklearn.manifold import TSNE
from umap import UMAP

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))

def visualize_pca(dataset: np.ndarray, title: str) -> None:
    pca = PCA(2)
    pca.fit(dataset)
    transformed_data = pca.transform(dataset)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title(title)
    plt.savefig(f"dr_plots/{title}.png")
    plt.close()

def visualize_autoencoder(dataset: np.ndarray, title: str) -> None:
    autoencoder = AutoEncoder(dataset.shape[1], 2)
    dataset_tensor = torch.tensor(dataset, dtype=torch.float32)
    projected_data = autoencoder.fit_transform(dataset_tensor).detach().numpy()
    plt.scatter(projected_data[:, 0], projected_data[:, 1])
    plt.title(title)
    plt.savefig(f"dr_plots/{title}.png")
    plt.close()

def visualize_tsne(dataset: np.ndarray, title: str, n_components=2, perplexity=30) -> None:
    tsne = TSNE(n_components=n_components, perplexity=perplexity)
    transformed_data = tsne.fit_transform(dataset)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title(title)
    plt.savefig(f"dr_plots/{title}.png")
    plt.close()

def visualize_umap(dataset: np.ndarray, title: str, n_components=2, min_dist=0.1, n_neighbours=15, metric='euclidean') -> None:
    umap = UMAP(n_components=n_components, min_dist=min_dist, n_neighbors=n_neighbours, metric=metric)
    transformed_data = umap.fit_transform(dataset)
    plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
    plt.title(title)
    plt.savefig(f"dr_plots/{title}.png")
    plt.close()


print("visualizing PCA...")
visualize_pca(dataset1, "PCA on Dataset 1")
visualize_pca(dataset2, "PCA on Dataset 2")

print("visualizing Autoencoder...")
visualize_autoencoder(dataset1, "Autoencoder on Dataset 1")
visualize_autoencoder(dataset2, "Autoencoder on Dataset 2")

print("visualizing t-SNE...")
visualize_tsne(dataset1, "t-SNE on Dataset 1 (Components 2, Perplexity=30)", perplexity=30)
visualize_tsne(dataset2, "t-SNE on Dataset 2 (Components 2, Perplexity=30)", perplexity=30)

print("visualizing t-SNE...")
visualize_tsne(dataset1, "t-SNE on Dataset 1 (Components 2, Perplexity=50)", perplexity=50)
visualize_tsne(dataset2, "t-SNE on Dataset 2 (Components 2, Perplexity=50)", perplexity=50)

print("visualizing UMAP...")

visualize_umap(dataset1, "UMAP on Dataset 1 (Distance cosine, Neighbors 15)", n_neighbours=15, metric='cosine')
visualize_umap(dataset2, "UMAP on Dataset 2 (Distance cosine, Neighbors 15)", n_neighbours=15, metric='cosine')

visualize_umap(dataset1, "UMAP on Dataset 1 (Distance euclidean, Neighbors 15)", n_neighbours=15, metric='euclidean')
visualize_umap(dataset2, "UMAP on Dataset 2 (Distance euclidean, Neighbors 15)", n_neighbours=15, metric='euclidean')



print("done!! :)")