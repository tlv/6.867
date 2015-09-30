import numpy as np
from hw1_res import homework1
import gd


def compute_Phi_poly(X, M):
  Phi = np.zeros((X.size, M+1))
  for i in range(M+1):
    Phi[:,i] = np.power(X,i)
  return Phi


def compute_Phi_trig(X, M):
  Phi = np.zeros((X.size, M+1))
  for i in range(M+1):
    Phi[:,i] = np.sin(2 * i * np.pi * X)
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

data = homework1.getData('curvefitting.txt')
X = data[0].reshape(data[0].size)
Y = data[1].reshape(data[1].size)
for M in [0,1,3,9]:
  Theta = max_likely(X,Y,M)
  Theta2 = np.zeros(Theta.size)
  Phi = compute_Phi_poly(X,M)
  print "-----------------"
  print sse(Theta, Phi, Y)
  print sse_der(Theta2, Phi, Y)
  print sse_der_2(Theta2, Phi, Y)


print compute_Phi_trig(X, 3)
