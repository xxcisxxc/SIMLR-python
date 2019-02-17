from LaplacianScore import LaplacianScore
import numpy as np
a = np.array([[1, 2, 3], [4, 2, 3], [6, 7, 10], [9, 6, 8]])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [1, 3, 5, 7], [2, 4, 6, 8]])
print(a)
print(b)
print(LaplacianScore(a, b))