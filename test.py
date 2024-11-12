import numpy as np
from Distance import Distance

x = np.array([1,2,3])
y = np.array([4,5,6])

# print(np.sqrt(np.sum((x - y) ** 2)))
# print(np.linalg.norm(x - y))

def other (x,y, p=2):
    return np.linalg.norm(x - y, ord=p) 





print(other(x,y), Distance.calculateMinkowskiDistance(x,y))



print(Distance.calculateMahalanobisDistance(x,y, np.eye(3)))
print(Distance.calculateCosineDistance(x,y))    
