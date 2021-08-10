# Higher order DGNet
# by Sølve Eidnes, 2021
# See "Order theory for discrete gradient methods" (https://arxiv.org/abs/2003.08267)
# 
# This code is based on
#   DGNet: Deep Energy-based Modeling of Discrete-Time Physics (2020)
#   by Takashi Matsubara, Ai Ishikawa, Takaharu Yaguchi
#   https://github.com/tksmatsubara/discrete-autograd
# which again builds on
#   Hamiltonian Neural Networks (2019)
#   by Sam Greydanus, Misko Dzamba, Jason Yosinski
#   https://github.com/greydanus/hamiltonian-nn

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_default_dtype(torch.float64)
import autograd
import autograd.numpy as np

import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from data_pend import get_dataset, dynamics_fn
from utils import L2_loss
import dgnet4

torch.set_default_dtype(torch.float64)

def train(args):
    torch.set_grad_enabled(False)

    # set random seed
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    # arrange data
    t_span = 20
    length = 40
    dt = t_span / (length - 1)
    data = get_dataset(seed=args['seed'], samples=50, noise_std=args['noise'], t_span=[0, t_span], timescale=length / t_span)
    train_x = torch.tensor(data['x'], requires_grad=True, dtype=torch.float64)
    test_x = torch.tensor(data['test_x'], requires_grad=True, dtype=torch.float64)
    train_dxdt = torch.tensor(data['dx'], dtype=torch.float64)
    test_dxdt = torch.tensor(data['test_dx'], dtype=torch.float64)
    x_reshaped = train_x.view(-1, length, args['input_dim'])
    x1 = x_reshaped[:, :-1].contiguous().view(-1, args['input_dim'])
    x2 = x_reshaped[:, 1:].contiguous().view(-1, args['input_dim'])
    dxdt = ((x2 - x1) / dt).detach()

    #### Training with Forward Euler:
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    # init model and optimizer
    if args['verbose']:
        print("Training with Forward Euler:")
    model_fe = dgnet4.DGNet(args['input_dim'], args['hidden_dim'],
                        nonlinearity=args['nonlinearity'], model=args['model'], solver=args['solver'])
    optim = torch.optim.Adam(model_fe.parameters(), args['learn_rate'], weight_decay=1e-4)
    for step in range(args['total_steps']+1):
        # train step
        with torch.enable_grad():
            dxdt_hat = model_fe.time_derivative(x1)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        test_dxdt_hat = model_fe.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        
        # logging
        if args['verbose'] and step % args['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    
    train_dxdt_hat = model_fe.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model_fe.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))

    #### Training with Runge-Kutta-4:
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    # init model and optimizer
    if args['verbose']:
        print("\nTraining with RK4:")
    #output_dim = 2
    model_rk4 = dgnet4.DGNet(args['input_dim'], args['hidden_dim'],
                        nonlinearity=args['nonlinearity'], model=args['model'], solver=args['solver'])
    #print('model.parameters(): ', model.parameters())
    optim = torch.optim.Adam(model_rk4.parameters(), args['learn_rate'], weight_decay=1e-4)
    for step in range(args['total_steps']+1):
        # train step
        with torch.enable_grad():
            dxdt_hat = model_rk4.discrete_time_derivative_rk4(x1, dt=dt)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        test_dxdt_hat = model_rk4.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        
        # logging
        if args['verbose'] and step % args['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    
    train_dxdt_hat = model_rk4.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model_rk4.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))  
    
    #### Training with 2nd order DG method:
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    # init model and optimizer
    if args['verbose']:
        print("\nTraining with 2nd order DG method:")
    model_dg = dgnet4.DGNet(args['input_dim'], args['hidden_dim'],
                        nonlinearity=args['nonlinearity'], model=args['model'], solver=args['solver'])
    optim = torch.optim.Adam(model_dg.parameters(), args['learn_rate'], weight_decay=1e-4)
    for step in range(args['total_steps']+1):
        # train step
        with torch.enable_grad():
            dxdt_hat = model_dg.discrete_time_derivative(x1, dt=dt, x2=x2)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        test_dxdt_hat = model_dg.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        
        # logging
        if args['verbose'] and step % args['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    
    train_dxdt_hat = model_dg.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model_dg.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))

    #### Training with 3rd order DG method:
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    # init model and optimizer
    if args['verbose']:
        print("\nTraining with 3rd order DG method:")
    model_dg3 = dgnet4.DGNet(args['input_dim'], args['hidden_dim'],
                        nonlinearity=args['nonlinearity'], model=args['model'], solver=args['solver'])
    optim = torch.optim.Adam(model_dg3.parameters(), args['learn_rate'], weight_decay=1e-4)
    for step in range(args['total_steps']+1):
        # train step
        with torch.enable_grad():
            dxdt_hat = model_dg3.discrete_time_derivative_dg3(x1, dt=dt, x2=x2)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        test_dxdt_hat = model_dg3.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        if args['verbose'] and step % args['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    
    train_dxdt_hat = model_dg3.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model_dg3.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))   

    #### Training with 4th order DG method:
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])
    # init model and optimizer
    if args['verbose']:
        print("\nTraining with 4th order DG method:")
    model_dg4 = dgnet4.DGNet(args['input_dim'], args['hidden_dim'],
                        nonlinearity=args['nonlinearity'], model=args['model'], solver=args['solver'])
    optim = torch.optim.Adam(model_dg4.parameters(), args['learn_rate'], weight_decay=1e-4)
    for step in range(args['total_steps']+1):
        # train step
        with torch.enable_grad():
            dxdt_hat = model_dg4.discrete_time_derivative_dg4(x1, dt=dt, x2=x2)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward() ; optim.step() ; optim.zero_grad()
        
        # run test data
        test_dxdt_hat = model_dg4.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)
        
        # logging
        if args['verbose'] and step % args['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))
    
    train_dxdt_hat = model_dg4.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model_dg4.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))  
    
    return model_fe, model_rk4, model_dg, model_dg3, model_dg4

if __name__ == "__main__":
    
    args = {}
    args['baseline'] = False
    args['feature'] = True
    args['field_type'] = 'solenoidal'
    args['hidden_dim'] = 200
    args['input_dim'] = 2
    args['learn_rate'] = .001
    args['name'] = 'pendulum'
    args['nonlinearity'] = 'tanh'
    args['print_every'] = 20
    args['seed'] = 0
    args['total_steps'] = 100
    args['use_rk4'] = False
    args['verbose'] = True
    args['model'] = 'hnn'
    args['solver'] = 'dg'
    args['noise'] = .1
    
    model_fe, model_rk4, model_dg, model_dg3, model_dg4 = train(args)
    
    #######################    
    
    y0 = np.array([2., 0.], dtype=np.float64)
    T = 10
    N = 100
    dt = T/N
    
    # Exact solution:
    t_span = [0,T]
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], 2000), 'rtol': 1e-12}
    pend_ivp = solve_ivp(fun=dynamics_fn, t_span=t_span, y0=y0, **kwargs)
    ys_exact = pend_ivp['y']
    
    # Integrating with solve_ivp:
    t_span = [0,T]
    kwargs = {'t_eval': np.linspace(0, T, int(1*N)), 'rtol': 1e-12}
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64).view(1,2)
        dx = model_fe.time_derivative(x).data.numpy().reshape(-1)
        return dx
    sol_ivp = solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    ys_fe = sol_ivp['y']
    m, l, g = 1, 1, 3
    energies_fe = 2*m*g*l*(1-np.cos(ys_fe[0,:])) + .5*l**2/m*ys_fe[1,:]**2
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64).view(1,2)
        dx = model_rk4.time_derivative(x).data.numpy().reshape(-1)
        return dx
    sol_ivp = solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    ys_rk4 = sol_ivp['y']
    energies_rk4 = 2*m*g*l*(1-np.cos(ys_rk4[0,:])) + .5*l**2/m*ys_rk4[1,:]**2
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64).view(1,2)
        dx = model_dg.time_derivative(x).data.numpy().reshape(-1)
        return dx
    sol_ivp = solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    ys_dg = sol_ivp['y']
    energies_dg = 2*m*g*l*(1-np.cos(ys_dg[0,:])) + .5*l**2/m*ys_dg[1,:]**2
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64).view(1,2)
        dx = model_dg3.time_derivative(x).data.numpy().reshape(-1)
        return dx
    sol_ivp = solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    ys_dg3 = sol_ivp['y']
    energies_dg3 = 2*m*g*l*(1-np.cos(ys_dg3[0,:])) + .5*l**2/m*ys_dg3[1,:]**2
    def fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64).view(1,2)
        dx = model_dg4.time_derivative(x).data.numpy().reshape(-1)
        return dx
    sol_ivp = solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)
    ys_dg4 = sol_ivp['y']
    energies_dg4 = 2*m*g*l*(1-np.cos(ys_dg4[0,:])) + .5*l**2/m*ys_dg4[1,:]**2
    
    
    fig = plt.figure(figsize=(4.5,4.5))
    plt.plot(ys_exact[0,:],ys_exact[1,:], 'k', label='Exact solution')  
    plt.plot(ys_fe[0,:],ys_fe[1,:], color=(1,0.7,0.3), label='Forward Euler')  
    plt.plot(ys_rk4[0,:],ys_rk4[1,:], color=(0.2,0.7,0.2), label='RK4')  
    plt.plot(ys_dg[0,:],ys_dg[1,:], color=(0.8,0,0.2), label='DGM2')   
    plt.plot(ys_dg3[0,:],ys_dg3[1,:], color=(0.2,0.8,.8), label='DGM3') 
    plt.plot(ys_dg4[0,:],ys_dg4[1,:], color=(0,0.4,1), label='DGM4')        
    plt.xlabel("$q$", fontsize=12)
    plt.ylabel("$p$", fontsize=12)
    plt.title("Comparison of training methods", fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig('training_pend.eps', format='eps', bbox_inches='tight')
    
    print('\nEnergy errors from integrating with solve_ivp from scipy:')
    fig = plt.figure(figsize=(5,4.5))
    plt.plot(np.linspace(0,T,N), np.abs(energies_fe-energies_fe[0]), color=(1,0.7,0.3), label='Forward Euler')
    plt.plot(np.linspace(0,T,N), np.abs(energies_rk4-energies_rk4[0]), color=(0.2,0.7,0.2), label='RK4')
    plt.plot(np.linspace(0,T,N), np.abs(energies_dg-energies_dg[0]), color=(0.8,0,0.2), label='DGM2')
    plt.plot(np.linspace(0,T,N), np.abs(energies_dg3-energies_dg3[0]), color=(0.2,0.8,.8), label='DGM3')  
    plt.plot(np.linspace(0,T,N), np.abs(energies_dg4-energies_dg4[0]), color=(0,0.4,1), label='DGM4')    
    plt.xlabel("$t$", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title("Energy errors", fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig('training_energy_pend.eps', format='eps', bbox_inches='tight')