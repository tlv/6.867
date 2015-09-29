import numpy as np
from numpy.linalg import inv

# x is numpy row array, y is numpy row array, l is lambda, M is degree of the basis
def ridge_regression(x, y, l, M):
	matrix_x = np.zeros(shape = (len(x), M + 1))
	for i in xrange(len(x)):
		matrix_x[i] = [x[i] ** j for j in xrange(M + 1)]
	identity = l * np.identity(M + 1)
	matrix_y = y[:, np.newaxis]
	return np.dot(np.dot(inv(np.dot(matrix_x.T, matrix_x) + identity), matrix_x.T), matrix_y)

xdata = np.array([0.000000, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.00000])
ydata = np.array([0.349486, 0.830839, 1.007332, 0.971507, 0.133066, 0.166823, -0.848307, -0.445686, -0.563567, 0.261502])
l = 0.5
M = 3
print ridge_regression(xdata, ydata, l, M)