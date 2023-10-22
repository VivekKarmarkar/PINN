# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 21:49:12 2023

@author: vkarmarkar
"""

import numpy as np
from scipy.optimize import minimize
import time
from Layer_basic_simplified import Identity


def build_layers(layers):
    num_layers = len(layers)
    
    for l in range(1,num_layers):
        layers[l].ni = layers[l-1].no
        layers[l].position = l
        layers[l]()
    
    layers[-1].name='output'
    layers[-1].activation = Identity
    
    return layers

def pack_params(layers,data='variables'):
    params = np.array([]) 
    if data == 'gradients' or data == 'loss':
        for layer in layers[1:]:
            params = np.append(params, layer.loss_weights.flatten())
            params = np.append(params, layer.loss_biases.flatten())
    elif data == 'variables':
        for layer in layers[1:]:
            params = np.append(params, layer.weights.flatten())
            params = np.append(params, layer.biases.flatten())        
    
    return params
            
def unpack_params(para,layers,data='variables'):

    ns = 0
    for layer in layers[1:]:
        W = para[ns: ns + layer.weights.size].reshape(layer.weights.shape)
        ns += layer.weights.size
        b = para[ns: ns + layer.biases.size]
        ns += layer.biases.size
        if data == 'variables':
            layer.weights = W
            layer.biases = b
        else:
            layer.weights_v = W
            layer.biases_v = b  

class PINN:
    
    def __init__(self,X,Y,layers,model_para=[],trainable_para=[]):
        
        if X.ndim == 1:
            X = X.reshape([-1,1])
            
        if len(Y) != 0 and Y.ndim == 1:
            Y = Y.reshape([-1,1])
            
        self.X = X
        self.Y = Y
        
        self.layers = build_layers(layers)    
        self.params = pack_params(self.layers)
        
        
    def __call__(self,x=[],init=True):
        if len(x) == 0:
            x = self.X
            init = False
            
        if x.ndim == 1:
            x = x.reshape([-1,1])               
        y = x
        for lay in self.layers:
            y = lay(y,init)
            
        return y

    def fit(self,method='BFGS',epoch=20, max_iter=100, tol=1e-6,learning_rate=0.01 ):
        
        if method == 'ADAM':
            self.train_ADAM(max_iter=max_iter,learning_rate=learning_rate, tol=tol)        
        else:
            self.train(method=method,max_iter=max_iter,epoch=epoch,tol=tol)
            
            
    def train(self,method,max_iter=1000,epoch=20,tol=1e-6):
        
        self.cost  = np.array([])
        self.iter  = np.array([])
        time_total = time.time()
        

        res0, loss_y, loss_yp, loss_para = self.loss_fun_eval()
        
        for i in range(1,epoch+1):
            time_start = time.time()
            if method == 'BFGS':
                res = minimize(self.loss_function, self.params, method='BFGS', jac=True, 
                                options={'maxiter': max_iter},tol=tol)
            elif method == 'NCG':
                res = minimize(self.loss_function, self.params, method='trust-ncg', jac=True, hessp=self.Hessp, 
                                options={'maxiter': max_iter},tol=tol)
            self.params = res.x
            self.cost = np.append(self.cost,res.fun)
            self.iter = np.append(self.iter,res.nit)
            time_elapsed = time.time() - time_start
            print( ' Epoch {:}. Elapsed time: {:1.5e}s. Loss: {:1.5e} '.format(i,time_elapsed, res.fun))
            
            if res.success or max(np.abs(res.jac))<tol or np.abs(res.fun-res0) < tol:
                break
            res0 = res.fun
            
        time_elapsed = time.time() - time_total
        print( ' Training completed. Elapsed time: {:1.5e}s Loss: {:1.5e}'.format(time_elapsed, res.fun))
        print( ' Number of iterations in each epoch: ', self.iter) 
    
    def loss_function(self,para):
        
        self.yp = np.array([])       
        unpack_params(para,self.layers)        
        loss, loss_y, loss_yp, loss_para = self.loss_fun_eval()

        jacobian = self.back_propagation(loss_y,loss_yp)
        
        return(loss, np.append(jacobian,loss_para))
    
    def loss_fun_eval(self):
        
        self.Y_pred = self(init=False)
        self.yp = np.array([])
        
        loss = 0.5*np.sum((self.Y_pred-self.Y)**2)/len(self.Y)
        loss_y = (self.Y_pred-self.Y)/len(self.Y)
        loss_yp = []    
        loss_para = []
        
        return loss, loss_y, loss_yp, loss_para
    
    def back_propagation(self,G,Gp=[]):
        
        for layer in self.layers[1:]:
            layer.loss_weights = np.zeros(layer.loss_weights.shape)
            layer.loss_biases = np.zeros(layer.loss_biases.shape) 

        n = len(self.layers)
        for l in range(n-1,0,-1):
            G = self.layers[l].back_prop_H(self.layers,G)
            
        jacobian = pack_params(self.layers,data='gradients')
        return jacobian
