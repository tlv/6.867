import numpy as np
from numpy.linalg import inv
from pylab import *

# x is numpy row array, y is numpy row array, l is lambda, M is degree of the basis
def ridge_regression(x, y, l, M):
	matrix_x = np.zeros(shape = (len(x), M + 1))
	for i in xrange(len(x)):
		matrix_x[i] = [x[i] ** j for j in xrange(M + 1)]
	print matrix_x
	identity = l * np.identity(M + 1)
	matrix_y = y[:, np.newaxis]
	return np.dot(np.dot(inv(np.dot(matrix_x.T, matrix_x) + identity), matrix_x.T), matrix_y)

xdata = np.array([0.000000, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.00000])
ydata = np.array([0.349486, 0.830839, 1.007332, 0.971507, 0.133066, 0.166823, -0.848307, -0.445686, -0.563567, 0.261502])
l = 1
M = 10
W = [i[0] for i in  ridge_regression(xdata, ydata, l, M)]

print W
rx = linspace(0, 1, 200)
ry = 0.44445594686119838 -0.32704035116221264 * rx -0.30870137298037709 * (rx ** 2) -0.16092522705038409 * (rx ** 3) -0.020195360919547531 * (rx ** 4) + 0.092642816811854004 * (rx ** 5)
plot(rx, ry)

x = [0.000000, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.00000]
y = [0.349486, 0.830839, 1.007332, 0.971507, 0.133066, 0.166823, -0.848307, -0.445686, -0.563567, 0.261502]
plot(x, y, 'bo', label = 'sampled')


grid(True)
savefig("test.png")
show()
