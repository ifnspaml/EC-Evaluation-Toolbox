from __future__ import division
# import tensorflow as tf
import numpy as np
import math
from scipy import linalg as LA

from matplotlib import pyplot as plt

class Kalman_filter():
  def __init__(self, M=1024, L=256, R=None, U=None, A = 0.998, alpha=None, beta = 0.5, overest = 1.5, fs = 16000, **kwargs):
    self.L = L
    if R is None:
      self.R = L #standard
    else: 
      self.R = R #low-latency
      assert R <= L
      assert L % R == 0
    if U is None:
      self.U = L
    else:
      self.U = U
    self.A = A
    if alpha is None:
      self.alpha = A
    else:
      self.alpha = alpha
    self.beta = beta
    self.Lambda = overest
    self.fs = fs

    self.M = M
    self.Nw = self.M-L
    self.f = L/self.M

    Fm = LA.dft(self.M)
    Fmi = np.conjugate(Fm)/self.M
    Q = np.concatenate((np.zeros((self.M-L,L)), np.eye(L)), 0)
    Qi = np.transpose(Q)

    self.G = Fm @ Q @ Qi @ Fmi

    self.win = np.hanning(2*self.R)
    self.win_sqrt = np.sqrt(self.win)
    self.win_s_pad = np.concatenate((self.win_sqrt, np.zeros((self.M-2*self.R,))))

    self.reset()

  def reset(self):
    self.W_est = np.zeros((self.M,),dtype='complex128')
    self.P = np.ones((self.M,))
    self.P_pred = np.ones((self.M,))
    self.Psi_delta = np.ones((self.M,))
    self.Psi_s = np.zeros((self.M,),dtype='complex128')
    self.Psi_s_last = np.zeros((self.M,),dtype='complex128')
    self.Psi_s_smooth = np.zeros((self.M,),dtype='complex128')
    self.eVec = np.zeros((self.R*2,))
    self.H = np.zeros((self.M,),dtype='complex128')

    self.oracle = False
    self.settings_dict = None

  def update(self, Y, X):
    self.Psi_delta = self.W_est * np.conj(self.W_est) + self.P
    self.Psi_delta *= (1-self.alpha**2)

    self.P_pred = self.alpha**2 * self.P + self.Psi_delta * self.Lambda

    self.W_est *= self.A

    E_tilde = Y - self.G @ (X * self.W_est)

    self.Psi_s = E_tilde * np.conj(E_tilde) + self.f * (X * self.P_pred * np.conj(X))

    self.Psi_s_smooth = (1-self.beta)*self.Psi_s + self.beta*self.Psi_s_last
    self.Psi_s_last = self.Psi_s
    self.Psi_s = self.Psi_s_smooth

    D = self.f * (X * self.P_pred * np.conj(X)) + self.Psi_s
    D_inv = 1/(D+np.finfo(float).eps)
    mu = self.f * self.P_pred * D_inv
    K = mu * np.conj(X)
    
    self.W_est += K * E_tilde

    self.P = self.P_pred - (self.f * K) * (X * self.P_pred)

  def enhance(self, Y, X):
    D_est = self.G @ (X*self.W_est)
    E = Y - D_est

    return E

  def run(self, y, x, reset_filter=True):

    if reset_filter:
      self.reset()

    n_frames = math.floor(len(x)/self.L)
    assert n_frames > 0
    fin_len = len(x)
    padding = fin_len % self.L
    if padding:
      x = np.concatenate((x, np.zeros(self.L-padding,)))
      y = np.concatenate((y, np.zeros(self.L-padding,)))

    e = np.zeros((len(x),))
    y_frame = np.zeros((self.M,))
    x_frame = np.zeros((self.M,))

    for m in range(0,len(x)-self.M,self.R):
      x_frame = np.roll(x_frame, -self.R)
      x_frame[-self.R:] = x[m:m+self.R]
      X = np.fft.fft(x_frame)

      y_frame = np.roll(y_frame, -self.R)
      y_frame[:self.M-self.L] = np.zeros((self.M-self.L,))
      y_frame[-self.R:] = y[m:m+self.R]
      Y = np.fft.fft(y_frame)

      if m % self.U == 0 and np.amax(np.abs(x_frame))>0:
        self.update(Y,X, D_real=D)

      E = self.enhance(Y,X)

      temp = np.fft.ifft(E)
      e[m:m+self.R] = temp[self.M-self.R:self.M] 

    if padding:
        e = e[:fin_len]

    return e
  
class NLMS_filter():
  def __init__(self, length, mu = 0.7, **kwargs):
    self.length = length
    self.state_size = [length,length]
    self.mu = mu
    self.delta = 1

    self.reset()

  def reset(self):
    self.filter = np.zeros((self.length,))
    self.reference = np.zeros((self.length,))

  def update_step(self, y, x):
    norm = np.dot(self.reference,self.reference)
    self.reference = np.roll(self.reference, shift=1, axis=-1)
    self.reference[0] = x

    d = np.dot(self.reference, self.filter)
    e_prio = y - d
    pre_factor = self.mu * e_prio / (norm + self.delta)
    self.filter += pre_factor * self.reference

  def prediction_step(self, y):
    d = np.dot(self.reference, self.filter)
    return y - d

  def call_step(self, y, x):

    self.update_step(y, x)
    output = self.prediction_step(y)

    return output

  def run(self, y,x):
    length = min(len(y),len(x))
    e = np.zeros((length,))
    for i in range(length):
      self.update_step(y[i], x[i])
      e[i] = self.prediction_step(y[i])

    return e