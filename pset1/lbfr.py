import numpy as np
from hw1_res import homework1
import pylab
import gd
import scipy.optimize as opt


def compute_Phi_poly(X, M):
  Phi = np.zeros((X.size, M+1))
  for i in range(M+1):
    Phi[:,i] = np.power(X,i)
  return Phi


def compute_Phi_trig(X, M):
  Phi = np.zeros((X.size, M+1))
  for i in range(M+1):
    Phi[:,i] = np.sin(2 * i * np.pi * X)
  Phi[:,0] = np.ones(Phi[:,0].size)
  return Phi


def max_likely(X, Y, M):
  Phi = compute_Phi_poly(X,M)
  return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(Phi), Phi)), np.transpose(Phi)), Y)


def sse(Theta, Phi, Y):
  Yhat = np.dot(Phi, Theta)
  return np.linalg.norm(Yhat - Y)**2


def sse_der(Theta, Phi, Y):
  return 2 * np.dot(np.transpose(Phi), np.dot(Phi, Theta) - Y)


def sse_der_2(Theta, Phi, Y):
  f = lambda x: sse(x, Phi, Y)
  return gd.gradient(f, Theta)

"""
data = homework1.getData('curvefitting.txt')
X = data[0].reshape(data[0].size)
Y = data[1].reshape(data[1].size)
pylab.plot(X, Y, 'bo')
for M in [3]:
  Theta = max_likely(X,Y,M)
  Theta2 = np.ones(Theta.size) * (10 ** 5)
  Phi = compute_Phi_trig(X,M)
  f = lambda theta: sse(theta, Phi, Y)
  fprime = lambda theta: sse_der(theta, Phi, Y)
  asdf = opt.fmin_bfgs(f, Theta2, full_output=True)
  Theta2 = asdf[0]
  print Theta2
  Xlin = np.linspace(0,1,200)
  Philin = compute_Phi_poly(Xlin, M)
  Phitrig = compute_Phi_trig(Xlin, M)
  Ylin_poly = np.dot(Philin, Theta)
  Ylin_sin = np.array([np.sin(2 * x * np.pi) for x in Xlin])
  Ylin_poly_2 = np.dot(Phitrig, Theta2)
  if M == 3:
    pylab.plot(Xlin, Ylin_poly, 'r')
    pylab.plot(Xlin, Ylin_sin, 'g')
    pylab.plot(Xlin, Ylin_poly_2, 'b')
pylab.show()
"""
