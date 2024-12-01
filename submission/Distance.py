import numpy as np
import math

class Distance:
    @staticmethod
    def calculateCosineDistance(x, y):
        return 1 - np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    @staticmethod
    def calculateMinkowskiDistance(x=None, y=None, p=2):
        if x is None or y is None:
            minkowski_wrapper = lambda x,y: np.sum(abs(x-y)**p)**(1/p) # TODO check overflow when p big
            return minkowski_wrapper
        return np.sum(abs(x-y)**p)**(1/p) # TODO check overflow when p big
    @staticmethod
    def calculateMahalanobisDistance(x,y, S_minus_1):
        delta = (x-y)
        return np.sqrt(np.dot(np.dot(delta.T, S_minus_1), delta))

