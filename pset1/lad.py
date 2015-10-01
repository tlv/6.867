import numpy as np
import gd
import lbfr
from hw1_res import homework1


def compute_lad(Phi, Theta, Y, lam=0):
  result = 0
  result += np.dot(np.ones(Y.size), np.absolute(np.dot(Phi, Theta) - Y))
  result += lam * np.linalg.norm(Theta)**2
  return result


def obj_func(Phi, Y, lam=0):
  f = lambda x: compute_lad(Phi, x, Y, lam=lam)
  return f


data_train = homework1.getData('regress_train.txt')
data_cv = homework1.getData('regress_validate.txt')
data_test = homework1.getData('regress_test.txt')

X_train = data_train[0].reshape(data_train[0].size)
Y_train = data_train[1].reshape(data_train[1].size)
X_cv = data_cv[0].reshape(data_cv[0].size)
Y_cv = data_cv[1].reshape(data_cv[1].size)
X_test = data_test[0].reshape(data_test[0].size)
Y_test = data_test[1].reshape(data_test[1].size)

mins = {}
asdf = 0
for M in range(10):
  for lam in [2 ** x for x in np.arange(-3, 6, 0.5)]:
    f = obj_func(lbfr.compute_Phi_poly(X_train, M), Y_train, lam)
    res = gd.gd(f, np.zeros(M+1), 0.001, 0.001, max_iters = 10000)
    Theta = res[0]
    mins[(M,lam)] = f(Theta)
    asdf += 1
    print asdf
print mins
