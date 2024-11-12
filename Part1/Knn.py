class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        # calculate distance between instance and all dataset
        distances = []
        for i in range(len(self.dataset)):
            distances.append(self.similarity_function(instance, self.dataset[i], self.similarity_function_parameters))
        
        # merge distances and labels
        distances = list(zip(distances, self.dataset_label))

        # sort distance and get the first K
        distances.sort(key= lambda x: x[0])
        distances = distances[:self.K]

        # get the most common label
        d = {}
        for _, label in distances:
            d[label] = d.get(label, 0) + 1
        
        return max(d, key=d.get)


