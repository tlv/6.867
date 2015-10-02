import numpy as np
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt
ep = 0.000001


def gd(f, start, step, cvg, max_iters=100000, print_vals=False):
  start = start.astype(float)
  iters = 1
  vals = [start]
  while iters < max_iters:
    grad = gradient(f, start)
    nextv = start - step * grad
    vals.append(nextv)
    if print_vals:
      print nextv
    if abs(f(start) - f(nextv)) < cvg:
      return nextv, iters, vals
    start = nextv
    iters += 1
  return nextv, iters, vals

def gradient(f, start):
  start = start.astype(float)
  start2 = np.copy(start)
  n = start.shape[0]
  plus = np.identity(n) * ep
  minus = np.identity(n) * (-ep)
  startd1 = np.tile(start, (n, 1))
  startd1 += plus
  startd2 = np.tile(start2, (n, 1))
  startd2 += minus
  plusv = np.array([f(startd1[i,:]) for i in range(n)])
  minusv = np.array([f(startd2[i,:]) for i in range(n)])
  return 1/(2*ep) * (plusv - minusv)
"""
def f1(x):
  return (x[0] - 2) ** 2 + (x[1] - 2) ** 2

def f2(x):
  return x[0]**2 + x[1]**2 + 4 * np.sin(x[0] + x[1])

guess = np.array([-10, -10])
cvg = 0.0000000000000001

print gd(f1, guess, 0.4, cvg)
a = gd(f2, guess, 0.2, cvg)
print a
print opt.fmin_bfgs(f1, guess, full_output=True)
print guess
print opt.fmin_bfgs(f2, guess, full_output=True)
plt.plot([x[0] for x in a[2]], [x[1] for x in a[2]])
plt.show()
"""
