import numpy as np
from numpy.linalg import inv
from pylab import *
import shelve

# x is numpy row array, y is numpy row array, l is lambda, M is degree of the basis
def ridge_regression(x, y, L, M):
	matrix_x = np.zeros(shape = (len(x), M + 1))
	for i in xrange(len(x)):
		matrix_x[i] = [x[i] ** j for j in xrange(M + 1)]
	identity = np.identity(M + 1)
	matrix_y = y[:, np.newaxis]
	return np.dot(np.dot(inv(np.dot(matrix_x.T, matrix_x) + L * identity), matrix_x.T), matrix_y)

def produce_plot(xdata, ydata, L, M, name):
	W = [i[0] for i in ridge_regression(xdata, ydata, L, M)]

	rx = linspace(min(xdata), max(xdata), 200)
	ry = 0
	for i in xrange(len(W)):
		ry += W[i] * (rx ** i)

	plot(rx, ry)
	plot(xdata, ydata, 'bo', label = 'sampled')
	grid(True)
	savefig(name)
	clf()

def error(x, y, L, theta, M):
	matrix_x = np.zeros(shape = (len(x), M + 1))
	for i in xrange(len(x)):
		matrix_x[i] = [x[i] ** j for j in xrange(M + 1)]
	matrix_y = y[:, np.newaxis]
	yhat = np.dot(matrix_x, theta)
	return np.linalg.norm(yhat - matrix_y)**2 + L * (np.dot(theta.T, theta))

def getData(name):
    data = loadtxt(name)
    X = data[0:1]
    Y = data[1:2]
    return X, Y

xdata = np.array([0.000000, 0.111111, 0.222222, 0.333333, 0.444444, 0.555556, 0.666667, 0.777778, 0.888889, 1.00000])
ydata = np.array([0.349486, 0.830839, 1.007332, 0.971507, 0.133066, 0.166823, -0.848307, -0.445686, -0.563567, 0.261502])
data_train = getData('regress_train.txt')
data_cv = getData('regress_validate.txt')
data_test = getData('regress_test.txt')
X_train = data_train[0][0]
Y_train = data_train[1][0]
X_cv = data_cv[0][0]
Y_cv = data_cv[1][0]
X_test = data_test[0][0]
Y_test = data_test[1][0]


def batch(src, x_data, y_data, Ms, Ls):
	results = {}
	for i in xrange(len(Ms)):
		for j in xrange(len(Ls)):
			M = Ms[i]
			L = Ls[j]
			name = "images/" + src + str(M) + "." + str(L) + ".png"
			print name
			produce_plot(x_data, y_data, L, M, name)
			theta = ridge_regression(x_data, y_data, L, M)
			results[(M, L)] = error(x_data, y_data, L, theta, M)
	return results


Ms = range(0, 10)
Ls = [0]
start = 10 ** - 10
for i in xrange(0, 11):
	Ls.append(start)
	start *= 10
Ls.append(2.0)
Ls.append(3.0)
Ls.append(4.0)
Ls.append(5.0)

# bishop = batch("bishop.", xdata, ydata, Ms, Ls)
# train = batch("train.", X_train, Y_train, Ms, Ls)

# d = shelve.open("training.db")
# d["train"] = train
# d.close() 

# d = shelve.open("training.db")
# train = d["train"]
# d.close()

# result = []
# for M in Ms:
# 	for L in Ls:
# 		result.append(((M, L), train[(M, L)][0][0]))
# result.sort(key = lambda x: x[1])
# print result

# validate = batch("validate.", X_cv, Y_cv, Ms, Ls)
# d = shelve.open("validate.db")
# d["validate"] = validate
# d.close()

d = shelve.open("validate.db")
validate = d["validate"]
d.close()

vresult = []
for M in Ms:
	for L in Ls:
		vresult.append(((M, L), validate[(M, L)][0][0]))
vresult.sort(key = lambda x: x[1])
print vresult


# # print error(xdata, ydata, L, theta)
# # produce_plot(xdata, ydata, L, M, name)

