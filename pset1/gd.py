import numpy as np
ep = 0.000001

def gd(f, start, step):
  pass

def gradient(f, start):
  start = start.astype(float)
  n = start.shape[0]
  plus = np.identity(n) * ep
  minus = np.identity(n) * (-ep)
  startd1 = np.tile(start, (n, 1))
  startd1 += plus
  startd2 = np.tile(start, (n, 1))
  startd2 += minus
  plusv = np.array([f(startd1[i,:]) for i in range(n)])
  minusv = np.array([f(startd2[i,:]) for i in range(n)])
  return 1/(2*ep) * (plusv - minusv)
