import numpy as np
import torch.nn as nn
import torch

class AutoEncoderNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        """
        Your autoencoder model definition should go here
        """
        first_hidden_layer = input_dim // 2
        second_hidden_layer = input_dim // 4

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, first_hidden_layer),
            nn.ReLU(),
            nn.Linear(first_hidden_layer, second_hidden_layer),
            nn.ReLU(),
            nn.Linear(second_hidden_layer, output_dim)
        )

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, second_hidden_layer),
            nn.ReLU(),
            nn.Linear(second_hidden_layer, first_hidden_layer),
            nn.ReLU(),
            nn.Linear(first_hidden_layer, input_dim),
            # nn.Sigmoid()
        )
    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function should map a given data matrix onto the bottleneck hidden layer
        :param x: the input data matrix of type torch.Tensor
        :return: the resulting projected data matrix of type torch.Tensor
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Your autoencoder model's forward pass operations should go here
        :param x: the input data matrix of type torch array
        :return: the neural network output as torch array
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AutoEncoder:

    def __init__(self, input_dim: int, projection_dim: int, learning_rate: float = 0.01, iteration_count: int = 500):
        """
        Initializes the Auto Encoder method
        :param input_dim: the input data space dimensionality
        :param projection_dim: the projection space dimensionality
        :param learning_rate: the learning rate for the auto encoder neural network training
        :param iteration_count: the number epoch for the neural network training
        """
        self.input_dim = input_dim
        self.projection_matrix = projection_dim
        self.iteration_count = iteration_count
        self.autoencoder_model = AutoEncoderNetwork(input_dim, projection_dim)
        """
            Your optimizer and loss definitions should go here
        """
        self.optimizer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss() # reconstruction loss

    def fit(self, x: torch.Tensor) -> None:
        """
        Trains the auto encoder nn on the given data matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should train the auto encoder to minimize the reconstruction error
        please do not forget to put the neural network model into the training mode before training
        """
        # put model into training mode
        self.autoencoder_model.train()

        for epoch in range(self.iteration_count):
            # reset gradients
            self.optimizer.zero_grad()
            # forward pass
            output = self.autoencoder_model(x)
            # calculate loss
            loss = self.loss(output, x)
            # backprop 
            loss.backward()
            # update weights
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{self.iteration_count}], Loss: {loss.item():.4f}")


    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        After training the nn a given dataset,
        this function uses the learned model to project new data instances
        :param x: the data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        please do not forget to put the neural network model into the evaluation mode before projecting data instances
        """
        # put model into evaluation mode
        self.autoencoder_model.eval()
        # project data but no need to calculate gradients
        with torch.no_grad():
            return self.autoencoder_model.project(x)
        
    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        # train the model and return the transformed data
        self.fit(x)
        return self.transform(x)
    

# test the implementation
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle

    dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))
    X = dataset2.astype(np.float32)
    print(X.shape, X.shape[1])

    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X)

    # Initialize and train the autoencoder
    autoencoder = AutoEncoder(input_dim=X.shape[1], projection_dim=2, learning_rate=0.01, iteration_count=500)
    autoencoder.fit(X_tensor)

    # Project data into 2D space
    X_projected = autoencoder.transform(X_tensor).numpy()

    # Visualize the 2D projection
    plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.7)
    plt.title("2D Projection using Autoencoder")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

    # # Visualize the 3D projection
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X_projected[:, 0], X_projected[:, 1], X_projected[:, 2], alpha=0.7)
    # ax.set_title("3D Projection using Autoencoder")
    # ax.set_xlabel("Dimension 1")
    # ax.set_ylabel("Dimension 2")
    # ax.set_zlabel("Dimension 3")
    # plt.show()

