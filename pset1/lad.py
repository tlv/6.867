import numpy as np
import scipy.optimize as opt
import lbfr
from hw1_res import homework1
import pylab


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
    res = opt.fmin_bfgs(f, np.zeros(M+1), full_output=True)
    Theta = res[0]
    mins[(M,lam)] = compute_lad(lbfr.compute_Phi_poly(X_cv, M), Theta, Y_cv, lam)

    if M == 6 and lam > 0.4 and lam < 0.55:
      Xlin = np.linspace(-3, 2, 500)
      Philin = lbfr.compute_Phi_poly(Xlin, M)
      Ylin_poly = np.dot(Philin, Theta)
      pylab.plot(Xlin, Ylin_poly, 'r')
      pylab.plot(X_train, Y_train, 'bo')
      pylab.plot(X_cv, Y_cv, 'ro')
      pylab.plot(X_test, Y_test, 'go')
min_k = None
for k in mins:
  if min_k == None or mins[k] < mins[min_k]:
    min_k = k
plist = []
for k in mins:
  plist.append((mins[k], k))
plist.sort()
for i in plist:
  print i
print min_k
print mins[min_k]
print X_train
pylab.show()
