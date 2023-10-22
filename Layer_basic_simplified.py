# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:34:44 2023

@author: vkarmarkar
"""

import numpy as np

def Identity(x):
    return x, np.ones(x.shape), np.zeros(x.shape)

def tanh(x):
    Phi = np.tanh(x)
    Phi_p = 1.0 - Phi**2
    Phi_pp = -2.0*Phi
    return Phi, Phi_p, Phi_pp

def xavier_init(size):
    np.random.seed(0)
    xavier_stddev = np.sqrt(2/sum(size))
    return np.random.normal(size = size, scale=xavier_stddev)

class Layer():
    
    def __init__(self,neurons,name='hidden',activation=tanh):

        if type(neurons) == int:
            self.no = neurons
        else:
            if len(neurons) == 2:
                (self.ni, self.no) = neurons
            else:
                self.no = neurons[0]

        self.activation = activation
        self.name = name

    def __call__(self,x=[],init=None):
        
        if len(x) == 0:
            ni = self.ni; no = self.no  

            self.biases = np.zeros(no)
            self.weights = xavier_init((ni,no))
            
            self.loss_biases = np.full(self.biases.shape,np.nan) # loss_biases reference to gradient wrt to b
            self.loss_weights = np.full(self.weights.shape,np.nan) # loss_biases reference to gradient wrt to weights

            return self
        
        else:
            H = x.dot(self.weights) + self.biases
            self.A, self.DA, self.DDA = self.activation(H)
            return self.A           
    
    def DA_to_DH(self,G):
        return self.DA*G
        
    def back_prop_H(self,layers,G):
   
        current_layer = self.position
        A = layers[current_layer-1].A
        self.loss_weights = np.dot(A.T, G)
        self.loss_biases = np.sum(G,axis=0)         
        G = np.dot(G,self.weights.T)
        G = layers[current_layer-1].DA_to_DH(G)
        
        return G
    
class Dense(Layer):
    pass

    
class Input(Layer):

    def __init__(self,neurons,name='input'):
        
        self.name = name
        self.no = neurons;
        self.ni = neurons
        self.activation = None
        self.weights = np.array([])
        self.biases = np.array([])
        self.loss_weights = np.array([])
        self.loss_biases = np.array([])
    
    def __call__(self,x=[],init=None):
        if len(x) == 0:
            pass            
        else:
            if x.ndim == 1:
                x = x.reshape([-1,1])
            self.A = x
            self.DA = np.ones(x.shape)
            self.dA = np.zeros(x.shape)
            npx, nf = x.shape
            self.D = np.asarray([np.eye(nf)]*npx)
            return x
