import numpy as np
from Part1.Distance import Distance
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.metrics import DistanceMetric
from scipy.spatial.distance import minkowski, mahalanobis, cosine


x = np.array([[1,2,3]])
y = np.array([[4,5,6]])

x1 = np.array([1,2,3])
y1 = np.array([4,5,6])

# print(np.sqrt(np.sum((x - y) ** 2)))
# print(np.linalg.norm(x - y))

# def other (x,y, p=2):
#     return np.linalg.norm(x - y, ord=p) 
# print(other(x,y), Distance.calculateMinkowskiDistance(x,y))
# print(Distance.calculateMahalanobisDistance(x,y, np.eye(3)))
# print(Distance.calculateCosineDistance(x,y))    


# ll = [(31,2), (33,100), (1,1)]
# ll.sort(key= lambda y: -y[0])
# print(ll)

# funs = [Distance.calculateCosineDistance, Distance.calculateMinkowskiDistance, Distance.calculateMahalanobisDistance]
# Sminus1 = np.eye(3)
# for fun in funs:
#     print(fun(x,y, Sminus1))

print(Distance.calculateCosineDistance(x1,y1))
print(cosine(x1,y1))

print(Distance.calculateMinkowskiDistance(x1,y1,21))
print(minkowski(x1,y1,21))

iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
print(Distance.calculateMahalanobisDistance(x1,y1, iv))
print(mahalanobis(x1,y1, iv))
