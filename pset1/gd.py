import numpy as np
import copy
ep = 0.1


def gd(f, start, step, cvg):
  start = start.astype(float)
  iters = 1
  while iters < 100000:
    grad = gradient(f, start)
    nextv = start - step * grad
    if abs(f(start) - f(nextv)) < cvg:
      return nextv, iters
    start = nextv
    iters += 1

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

def square(x):
  return (x - 3) ** 2

print gd(square, np.array([0]), 0.2, 0.00000001)