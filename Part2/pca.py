import numpy as np

class PCA:
    def __init__(self, projection_dim: int):
        """
        Initializes the PCA method
        :param projection_dim: the projection space dimensionality
        """
        self.projection_dim = projection_dim
        # keeps the projection matrix information
        self.projection_matrix = None
        self.mean = None

    def fit(self, x: np.ndarray) -> None:
        """
        Applies the PCA method and obtains the projection matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should assign the resulting projection matrix to self.projection_matrix
        """
        # calculate mean of each column
        self.mean = np.mean(x, axis=0)
        # center columns by subtracting column means
        centered_data = x - self.mean
        # calculate covariance matrix of centered matrix
        covariance_matrix = np.cov(centered_data.T) # since np.cov expects features as rows
        # eigenvector and eigenvalues
        eig_values, eig_vectors = np.linalg.eig(covariance_matrix)

        # sort eigenvectors based on eigenvalues
        idx = eig_values.argsort()[::-1] # descending order
        eig_values = eig_values[idx]
        eig_vectors = eig_vectors[:, idx] # sinece eigenvectors are column wise
        eig_values = eig_values.real
        eig_vectors = eig_vectors.real

        # select the first k eigenvectors
        self.projection_matrix = eig_vectors[:, :self.projection_dim]


    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        After learning the projection matrix on a given dataset,
        this function uses the learned projection matrix to project new data instances
        :param x: data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        this function should utilize self.projection_matrix for the operations
        """
        # center columns by subtracting column means
        centered_data = x - self.mean
        # project data
        return np.dot(centered_data, self.projection_matrix)
    

# test the implementation
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris

    data = load_iris()

    X = data.data
    y = data.target

    pca = PCA(2)

    pca.fit(X)
    X_pca = pca.transform(X)

    print(f"shape of X: {X.shape}")
    print(f"shape of X_pca: {X_pca.shape}")

    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
    plt.show()