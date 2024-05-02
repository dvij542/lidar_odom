""" Train an ELM model for tires and alo predict aerodynamic parameters given the data
"""

__author__ = 'Dvij Kalaria'
__email__ = 'dkalaria@andrew.cmu.edu'


import sys

# print(sys.path)
sys.path.append('./')
import time
import numpy as np
import matplotlib.pyplot as plt
import math

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from sklearn.pipeline import make_pipeline
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from bayes_race.params import ORCA, CarlaParams, RCParams
from bayes_race.models import Kinematic6, Dynamic
import random
from bayes_race.utils.plots import plot_true_predicted_variance
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

#####################################################################
# load data

parser = argparse.ArgumentParser(description='Arguments for offline fitting of tire curves')
# parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240302-143549-1_data.npz',help='Specify the file path')
# parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240306-001713_data.npz',help='Specify the file path')
parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240316-024251_data.npz',help='Specify the file path')
# parser.add_argument('--file', required=False, default='processed_data/vicon-circle-data-20240225-215833_data.npz',help='Specify the file path')
parser.add_argument('--model_path', required=False, default='orca/semi_mlp-v1.pickle',help='Specify the model save file path')
parser.add_argument('--dt', required=False, default=0.05,help='Specify the time step')
parser.add_argument('--ignore_first', required=False, default=10,help='Specify the no of initial time steps to ignore')
parser.add_argument('--min_v', required=False, default=1.,help='Min velocity for masking out')
parser.add_argument('--max_v', required=False, default=3.5,help='Max velocity for masking out')
parser.add_argument('--resolution', required=False, default=1,help='Resolution for finite difference')
parser.add_argument('--n_iters', required=False, default=2000,help='No of iters for training')
parser.add_argument('--save', required=False, action='store_true',help='Whether to save trained model')
parser.add_argument('--seed', required=False, default=1,help='seed for random number generator')

# Parse the command-line arguments
args = parser.parse_args()

print(args.file)
SAVE_MODELS = args.save
MODEL_PATH = args.model_path
N_ITERS = int(args.n_iters)
FILE_NAME = args.file
RES = int(args.resolution)
ignore_first = args.ignore_first
min_v = float(args.min_v)
max_v = float(args.max_v)

state_names = ['x', 'y', 'yaw', 'vx', 'vy', 'omega']

torch.manual_seed(args.seed)
random.seed(0)
np.random.seed(0)

alpha_f_distribution_y = np.zeros(2000)
alpha_f_distribution_x = np.arange(-1.,1.,2./2000)

alpha_r_distribution_y = np.zeros(2000)
alpha_r_distribution_x = np.arange(-1.,1.,2./2000)

class ResidualModel(torch.nn.Module):
    def __init__(self, model, deltat = args.dt):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.Rx = torch.nn.Linear(1,1).to(torch.float64)
        self.Rx.weight.data.fill_(0.)
        
        self.Ry = torch.nn.Linear(1,1,bias=False).to(torch.float64)
        self.Ry.weight.data.fill_(40.)
        
        self.Fy = torch.nn.Linear(1,1,bias=False).to(torch.float64)
        self.Fy.weight.data.fill_(40.)
        
        self.deltat = deltat
        self.model = model
        self.b = torch.arange(0.,.6,(0.6)/12.).to(torch.float64).unsqueeze(0)

    
    def get_force_F(self,alpha_f) :
        return self.Fy(alpha_f)
        
    def get_force_R(self,alpha_r) :
        return self.Ry(alpha_r)
        
    def forward(self, x, debug=False,n_divs=1):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        # print(x.shape)
        # out = X
        pitch = torch.tensor(0.)#x[:,7]
        theta = x[:,2]
        pwm = x[:,0]
        out = torch.zeros_like(x[:,3:6])
        for i in range(n_divs) :
            vx = (x[:,3] + out[:,0]).unsqueeze(1)
            vy = x[:,4] + out[:,1]
            w = x[:,5] + out[:,2]
            alpha_f = (theta - torch.atan2(w*self.model.lf+vy,vx[:,0])).unsqueeze(1)
            alpha_r = torch.atan2(w*self.model.lr-vy,vx[:,0]).unsqueeze(1)
            if debug :
                for alpha in alpha_f[:,0] :
                    alpha_f_distribution_y[int((alpha+1.)*1000)] += 1
                for alpha in alpha_r[:,0] :
                    alpha_r_distribution_y[int((alpha+1.)*1000)] += 1
            Ffy = self.Fy(alpha_f)[:,0]
            Fry = self.Ry(alpha_r)[:,0]
            Frx = self.Rx(vx)[:,0]
            # a_pred = (pwm>0)*self.model.Cm1*pwm*(3.45*0.919)/(0.34*1265) \
            #     + (pwm<=0)*self.model.Cm2*pwm*(3.45*0.919)/(0.34*1265)
            # Frx_kin = (self.model.Cm1-self.model.Cm2*vx[:,0])*pwm
            Frx = 0.#a_pred + self.model.Cm1*pwm
            vx_dot = (Frx-Ffy*torch.sin(theta)+vy*w-9.8*torch.sin(pitch))
            vy_dot = (Fry+Ffy*torch.cos(theta)-vx[:,0]*w)
            # print(self.model.mass/self.model.Iz)
            w_dot = (Ffy*self.model.lf*torch.cos(theta)-Fry*self.model.lr)/(4.*0.15*0.15)
            out += torch.cat([vx_dot.unsqueeze(dim=1),vy_dot.unsqueeze(dim=1),w_dot.unsqueeze(dim=1)],axis=1)*self.deltat/n_divs
        out2 = (out)
        return out2


def load_data(file_name):
    data_dyn = np.load(file_name)
    y_all = (data_dyn['states'].T[3:6,1:]).T
    # print(data_dyn['inputs'].shape)
    # print(data_dyn['states'].shape)
    x = np.concatenate([
        data_dyn['inputs'].T[:,:-1].T,
        data_dyn['states'].T[3:6,:-1].T],
        axis=1)
    mask = x[:,-1:]
    return torch.tensor(x), torch.tensor(y_all), torch.tensor(mask)
    

x_train, y_train, mask = load_data(FILE_NAME)

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = SimpleModel(5,10,3)
# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=.2)
model = model.double()

def train_step(states,cmds,augment=True) :
    global model
    X = np.concatenate((np.array(states),np.array(cmds)),axis=1)[:-1]
    Y = (np.array(states)[1:] - np.array(states)[:-1])
    
    
    # Augmentation
    if augment :
        X_ = X.copy()
        X_[:,1] = -X[:,1]
        X_[:,2] = -X[:,2]
        X_[:,4] = -X[:,4]
        Y_ = Y.copy()
        Y_[:,1] = -Y[:,1]
        Y_[:,2] = -Y[:,2]
        
        X = np.concatenate((X,X_),axis=0)    
        Y = np.concatenate((Y,Y_),axis=0)
    
        X_ = X.copy()
        X_[:,1] = 0.
        X_[:,2] = 0.
        X_[:,4] = 0.
        Y_ = Y.copy()
        Y_[:,1] = 0.
        Y_[:,2] = 0.
        
        X = np.concatenate((X,X_),axis=0)
        Y = np.concatenate((Y,Y_),axis=0)
    
    X = torch.tensor(X).double()
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X)
    
    Y = torch.tensor(Y)
    # Compute the loss
    loss = criterion(outputs, Y)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()    
    
    return

#####################################################################
# load vehicle parmaeters (Use only geometric parameters)

params = RCParams(control='pwm')
vehicle_model = Dynamic(**params)


start = time.time()

#####################################################################
# Train the model

# Optimizers specified in the torch.optim package
model.train()
# model.cuda()
t1 = time.time()
pred_w = []
pred_w_ = []
_pred_w = []
ac_w = []
kin_w = []
steers = []
for i in range(ignore_first,len(x_train)-100) :
    # Zero your gradients for every batch!
    # print(x_train.shape)
    train_step(x_train[i:i+100,2:],x_train[i:i+100,:2],augment=True) 
    vx = x_train[i+100,2]
    steer = x_train[i+100,1]
    steers.append(4*steer)
    L = 0.36
    kin_w.append(vx*np.tan(steer)/L)
    ac_w.append(y_train[i+100,2])
    X_ = x_train[i+100,[2,3,4,0,1]].unsqueeze(0)
    for j in range(5) :
        X_[:,:3] += model(X_)
    pred_w.append((X_[:,:3]).detach().numpy()[0,2])
    X_ = x_train[i+100,[2,3,4,0,1]].unsqueeze(0)
    X_[:,-1] *= 1.5 #0.#1.5
    for j in range(5) :
        X_[:,:3] += model(X_)
    pred_w_.append((X_[:,:3]).detach().numpy()[0,2])
    X_ = x_train[i+100,[2,3,4,0,1]].unsqueeze(0)
    X_[:,-1] *= -1 #0.#1.5
    for j in range(20) :
        X_[:,:3] += model(X_)
    _pred_w.append((X_[:,:3]).detach().numpy()[0,2])
t2 = time.time()

model.eval()
plt.plot(kin_w,label='kinematic')
plt.plot(steers,label='steer')
plt.plot(ac_w,label='actual')
plt.plot(pred_w,label='predicted')
plt.plot(pred_w_,label='predicted_')
plt.plot(_pred_w,label='predicted__')
plt.legend()
plt.show()


if SAVE_MODELS :
    torch.save(model.state_dict(), MODEL_PATH)
