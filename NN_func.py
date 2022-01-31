import numpy as np
import pandas as pd
import torch
import h5py
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,p_drop=0.3):
        super(Feedforward, self).__init__()
        
        # init parameters
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_layers  = num_layers
        
        # network related torch stuffs
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=p_drop)

        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, self.hidden_size)])
        self.linears.extend([torch.nn.Linear(self.hidden_size, self.hidden_size) for i in range(1, self.num_layers)])
      
        # output Y1
        self.out = torch.nn.Linear(self.hidden_size, 1) # Linear output
    
    def Y1_latent_pred(self,x):
        """core neural network"""
        for layer in self.linears: # Feedforward
            x = self.dropout(self.relu(layer(x)))
        return self.out(x)[:,0]
    
    def Y2_f(self, x):
        """the known Y2 function"""
        return -4.5 + 2.0*x
    
    def forward(self, x):     
        return self.Y2_f(self.Y1_latent_pred(x))
       




def open_data(vec=False, sc=False) :
    
    with h5py.File('data_train.hdf5', 'r') as f :
        #for key in f.keys():
        #    print(key)
        d1 = f['xtrain']
        d2 = f['ytrain']
        d3 = f['xtrain_sc']
        d4 = f['ytrain_sc']
        X_train = d1[:] 
        y_train = d2[:]
        X_train_sc = d3[:]
        y_train_sc = d4[:]
    
    with h5py.File('data_valid.hdf5', 'r') as f :
        #for key in f.keys():
        #    print(key)
        d1 = f['xvalid']
        d2 = f['yvalid']
        d3 = f['xvalid_sc']
        d4 = f['yvalid_sc']
        X_valid = d1[:] 
        y_valid = d2[:]
        X_valid_sc = d3[:]
        y_valid_sc = d4[:]
    
    with h5py.File('data_test.hdf5', 'r') as f :
        #for key in f.keys():
        #    print(key)
        d1 = f['xtest']
        d2 = f['ytest']
        d3 = f['xtest_sc']
        d4 = f['ytest_sc']
        X_test = d1[:] 
        y_test = d2[:]
        X_test_sc = d3[:]
        y_test_sc = d4[:]
    
                
    if vec == True :
        
        if sc == False :
        
            return X_train, y_train, X_valid, y_valid, X_test, y_test
        
        if sc == True :
        
            return X_train_sc, y_train_sc, X_valid_sc, y_valid_sc, X_test_sc, y_test_sc
   
    else :
        
        datas_train = {"X" : torch.FloatTensor(X_train), "y" : torch.FloatTensor(y_train), "X_sc" : torch.FloatTensor(X_train_sc),
                      "y_sc" : torch.FloatTensor(y_train_sc)}

        datas_valid = {"X" : torch.FloatTensor(X_valid), "y" : torch.FloatTensor(y_valid), "X_sc" : torch.FloatTensor(X_valid_sc),
                      "y_sc" : torch.FloatTensor(y_valid_sc)}

        datas_test = {"X" : torch.FloatTensor(X_test), "y" : torch.FloatTensor(y_test), "X_sc" : torch.FloatTensor(X_test_sc),
                      "y_sc" : torch.FloatTensor(y_test_sc)}
        
        return datas_train, datas_valid, datas_test
    
 

    
