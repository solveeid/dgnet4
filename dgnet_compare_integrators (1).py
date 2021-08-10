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
import autograd.numpy as np

import numpy.linalg as la
import time

import matplotlib.pyplot as plt
import scipy.integrate
solve_ivp = scipy.integrate.solve_ivp

import os
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from data_spring import get_dataset
from utils import L2_loss
import dgnet4

torch.set_default_dtype(torch.float64)

def train(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = torch.get_default_dtype()
    torch.set_grad_enabled(False)

    # set random seed
    torch.manual_seed(args['seed'])
    np.random.seed(args['seed'])

    # init model and optimizer
    model = dgnet4.DGNet(args['input_dim'], args['hidden_dim'],
                        nonlinearity=args['nonlinearity'], model=args['model'], solver=args['solver'])
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), args['learn_rate'], weight_decay=1e-4)

    # arrange data
    t_span = 20
    length = 100
    dt = t_span / (length - 1)
    data = get_dataset(seed=args['seed'], noise_std=args['noise'], t_span=[0, t_span], timescale=length / t_span)
    train_x = torch.tensor(data['x'], requires_grad=True, device=device, dtype=dtype)
    test_x = torch.tensor(data['test_x'], requires_grad=True, device=device, dtype=dtype)
    train_dxdt = torch.tensor(data['dx'], device=device, dtype=dtype)
    test_dxdt = torch.tensor(data['test_dx'], device=device, dtype=dtype)

    input_dim = train_x.shape[-1]
    x_reshaped = train_x.view(-1, length, input_dim)
    x1 = x_reshaped[:, :-1].contiguous().view(-1, input_dim)
    x2 = x_reshaped[:, 1:].contiguous().view(-1, input_dim)
    dxdt = ((x2 - x1) / dt).detach()

    # vanilla train loop
    for step in range(args['total_steps'] + 1):
        with torch.enable_grad():
            # train step
            dxdt_hat = model.discrete_time_derivative(x1, dt=dt, x2=x2)
            loss = L2_loss(dxdt, dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()

        # run test data
        test_dxdt_hat = model.time_derivative(test_x)
        test_loss = L2_loss(test_dxdt, test_dxdt_hat)

        # logging
        if args['verbose'] and step % args['print_every'] == 0:
            print("step {}, train_loss {:.4e}, test_loss {:.4e}".format(step, loss.item(), test_loss.item()))

    train_dxdt_hat = model.time_derivative(train_x)
    train_dist = (train_dxdt - train_dxdt_hat)**2
    test_dxdt_hat = model.time_derivative(test_x)
    test_dist = (test_dxdt - test_dxdt_hat)**2
    print('Final train loss {:.4e}\nFinal test loss {:.4e}'
          .format(train_dist.mean().item(), test_dist.mean().item()))

    return model

if __name__ == "__main__":
    
    args = {}
    args['baseline'] = False
    args['feature'] = True
    args['field_type'] = 'solenoidal'
    args['hidden_dim'] = 200
    args['input_dim'] = 2
    args['learn_rate'] = .001
    args['name'] = 'spring'
    args['nonlinearity'] = 'tanh'
    args['print_every'] = 20
    args['seed'] = 0
    args['total_steps'] = 600
    args['use_rk4'] = False
    args['verbose'] = True
    args['model'] = 'hnn'
    args['solver'] = 'dg'
    args['noise'] = .1
    
    print('Training the model with the second order discrete gradient method:')
    model = train(args)
    
    #######################
    
    def newton_DGM(u,un,dt,tol):
        S = np.array([[0, 1],[-1, 0]])
        I = np.array([[1, 0],[0, 1]])
        def fun(y0, y1, dt):
            y0 = torch.tensor(y0, requires_grad=True, dtype=torch.float64).view(1,2)
            y1 = torch.tensor(y1, requires_grad=True, dtype=torch.float64).view(1,2)
            return (1/dt*(y1.detach().numpy() - y0.detach().numpy()) - np.dot(S,model.grad(x1=y0,x2=y1).detach().numpy().T).T)[0]
        f = lambda un: fun(u,un,dt)
        def fun2(y0, y1, dt):
            y0 = torch.tensor(y0, requires_grad=True, dtype=torch.float64).view(1,2)
            y1 = torch.tensor(y1, requires_grad=True, dtype=torch.float64).view(1,2)
            return 1/dt*I - np.dot(S,model.jacgrad(x1=y0,x2=y1)[0].detach().numpy().T)
        J = lambda un: fun2(u,un,dt)
        err = la.norm(f(un))
        c = 0
        while err > tol:
            un = un - la.solve(J(un),f(un))
            err = la.norm(f(un))
            c = c+1
            if c > 5:
                break
        return un
    
    def newton_DGM3(y0,yn,dt,tol):
        S = np.array([[0, 1],[-1, 0]])
        I = np.array([[1, 0],[0, 1]])
        y0 = torch.tensor(y0, requires_grad=True, dtype=torch.float64).view(1,2)
        z = (y0.detach().numpy() + 2/3*dt*np.dot(S,model.grad(x1=y0).detach().numpy().T).T)[0]
        z = torch.tensor(z, requires_grad=True, dtype=torch.float64).view(1,2)
        HH = model.jacgrad(y0)[0] # the Hessian of H at y0
        SH = np.dot(S,HH.numpy())
        SHS = np.dot(SH,S)
        SHSHS = np.dot(SH,SHS)
        DDGz = model.jacgrad(y0,z)[0] # Jacobian of discrete gradient DGH(y0,z)  w.r.t. z
        Q = .5*(DDGz.numpy().T-DDGz.numpy())
        SQS = np.dot(S,np.dot(Q,S))
        St = S + dt*SQS - 1/12*dt**2*SHSHS
        def f(y1):
            y1 = torch.tensor(y1, requires_grad=True, dtype=torch.float64).view(1,2)
            return (1/dt*(y1.detach().numpy() - y0.detach().numpy()) - np.dot(St,model.grad(x1=y0,x2=y1).detach().numpy().T).T)[0]
        def J(y1):
            y1 = torch.tensor(y1, requires_grad=True, dtype=torch.float64).view(1,2)
            return 1/dt*I - np.dot(St,model.jacgrad(x1=y0,x2=y1)[0].detach().numpy().T)
        err = la.norm(f(yn))
        c = 0
        while err > tol:
            yn = yn - la.solve(J(yn),f(yn))
            err = la.norm(f(yn))
            c = c+1
            if c > 5:
                break
        return yn
    
    def newton_DGM4(y0,yn,dt,tol):
        S = np.array([[0, 1],[-1, 0]])
        I = np.array([[1, 0],[0, 1]])
        y0 = torch.tensor(y0, requires_grad=True, dtype=torch.float64).view(1,2)
        yh = (y0.detach().numpy() + .5*dt*np.dot(S,model.grad(x1=y0).detach().numpy().T).T)[0]
        yh = torch.tensor(yh, requires_grad=True, dtype=torch.float64).view(1,2)
        z = (y0.detach().numpy() + .75*dt*np.dot(S,model.grad(x1=yh).detach().numpy().T).T)[0]
        z = torch.tensor(z, requires_grad=True, dtype=torch.float64).view(1,2)
        HH = model.jacgrad(yh)[0] # the Hessian of H at yh
        SH = np.dot(S,HH.numpy())
        SHS = np.dot(SH,S)
        SHSHS = np.dot(SH,SHS)
        DDGz = model.jacgrad(y0,z)[0] # Jacobian of discrete gradient DGH(y0,z)  w.r.t. z
        Q = .5*(DDGz.numpy().T-DDGz.numpy())
        SQS = np.dot(S,np.dot(Q,S))
        St = S + 8/9*dt*SQS - 1/12*dt**2*SHSHS
        def f(y1):
            y1 = torch.tensor(y1, requires_grad=True, dtype=torch.float64).view(1,2)
            return (1/dt*(y1.detach().numpy() - y0.detach().numpy()) - np.dot(St,model.grad(x1=y0,x2=y1).detach().numpy().T).T)[0]
        def J(y1):
            y1 = torch.tensor(y1, requires_grad=True, dtype=torch.float64).view(1,2)
            return 1/dt*I - np.dot(St,model.jacgrad(x1=y0,x2=y1)[0].detach().numpy().T)
        err = la.norm(f(yn))
        c = 0
        while err > tol:
            yn = yn - la.solve(J(yn),f(yn))
            err = la.norm(f(yn))
            c = c+1
            if c > 5:
                break
        return yn
    
    ##########
    
    y0 = np.array([1., 0.], dtype=np.float64)
    T = 50
    N = 500
    dt = T/N
    
    print('\nIntegrating until time', T, 'with', N, 'steps, for each of the different schemes:')
    
    # Integrate with RK4:
    tic = time.time()
    ys = np.zeros([y0.shape[0], N])
    ys[:,0] = y0
    y = y0
    energies = np.zeros(N)
    energies[0] = model.hamiltonian(torch.tensor(y0)).detach().numpy()[0]
    def f(np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float64).view(1,2)
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx
    for i in range(N):
        yn = y + dt*1/6*(f(y)+2*f(y+.5*dt*f(y))+2*f(y+.5*dt*f(y+.5*dt*f(y)))+f(y+dt*f(y+.5*dt*f(y+.5*dt*f(y)))));
        energies[i] = model.hamiltonian(torch.tensor(yn)).detach().numpy()[0]
        y = yn
        ys[:,i] = y
    print('The RK4 scheme used {:f} seconds.'.format(time.time()-tic))
    ys_rk4 = ys
    energies_rk4 = energies
    
    # Integrate with DGM:
    tic = time.time()
    ys = np.zeros([y0.shape[0], N])
    ys[:,0] = y0
    y = y0
    energies = np.zeros(N)
    energies[0] = model.hamiltonian(torch.tensor(y0)).detach().numpy()[0]
    for i in range(N):
        yn = newton_DGM(y,y,dt,1e-16).astype(np.float64)
        energies[i] = model.hamiltonian(torch.tensor(yn)).detach().numpy()[0]
        y = yn
        ys[:,i] = y
    print('The DG scheme used {:f} seconds.'.format(time.time()-tic))
    ys_dgm = ys
    energies_dgm = energies
    
    # Integrate with DGM3:
    tic = time.time()
    ys = np.zeros([y0.shape[0], N])
    ys[:,0] = y0
    y = y0
    energies = np.zeros(N)
    energies[0] = model.hamiltonian(torch.tensor(y0)).detach().numpy()[0]
    for i in range(N):
        yn = newton_DGM3(y,y,dt,1e-16).astype(np.float64)
        energies[i] = model.hamiltonian(torch.tensor(yn)).detach().numpy()[0]
        y = yn
        ys[:,i] = y
    print('The DG3 scheme used {:f} seconds.'.format(time.time()-tic))
    ys_dgm3 = ys
    energies_dgm3 = energies
    
    # Integrate with DGM4:
    tic = time.time()
    ys = np.zeros([y0.shape[0], N])
    ys[:,0] = y0
    y = y0
    energies = np.zeros(N)
    energies[0] = model.hamiltonian(torch.tensor(y0)).detach().numpy()[0]
    for i in range(N):
        yn = newton_DGM4(y,y,dt,1e-16).astype(np.float64)
        energies[i] = model.hamiltonian(torch.tensor(yn)).detach().numpy()[0]
        y = yn
        ys[:,i] = y
    print('The DG4 scheme used {:f} seconds.'.format(time.time()-tic))
    ys_dgm4 = ys
    energies_dgm4 = energies
    
    fig = plt.figure(figsize=(5,4.5))  
    plt.plot(ys_rk4[0,:],ys_rk4[1,:], color=(0.2,0.7,0.2), label='RK4')  
    plt.plot(ys_dgm[0,:],ys_dgm[1,:], color=(0.8,0,0.2), label='DGM2')     
    plt.plot(ys_dgm3[0,:],ys_dgm3[1,:], color=(0.2,0.8,.8), label='DGM3')     
    plt.plot(ys_dgm4[0,:],ys_dgm4[1,:], color=(0,0.4,1), label='DGM4')        
    plt.xlabel("$q$", fontsize=12)
    plt.ylabel("$p$", fontsize=12)
    plt.title("Comparison of solvers", fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig('solvers_spring.eps', format='eps', bbox_inches='tight')
    
    fig = plt.figure(figsize=(5,4.5))
    plt.semilogy(np.linspace(0,T,N), np.abs(energies_rk4-energies_rk4[0]), color=(0.2,0.7,0.2), label='RK4')   
    plt.semilogy(np.linspace(0,T,N), np.abs(energies_dgm-energies_dgm[0]), color=(0.8,0,0.2), label='DGM2')
    plt.semilogy(np.linspace(0,T,N), np.abs(energies_dgm3-energies_dgm3[0]), color=(0.2,0.8,.8), label='DGM3')
    plt.semilogy(np.linspace(0,T,N), np.abs(energies_dgm4-energies_dgm4[0]), color=(0,0.4,1), label='DGM4')   
    plt.xlabel("$t$", fontsize=12)
    plt.ylabel("Error", fontsize=12)
    plt.title("Energy errors", fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig('energyerrors_spring.eps', format='eps', bbox_inches='tight')
    
    ##### Order plot:
    print('\nGenerating order plots:')
    maxit = 7
    mn = 10
    T = 1
    
    # Reference solution:
    print('Calculating reference solution...')
    N = mn*(2**maxit+1)
    dt = T/N
    y = y0
    S = np.array([[0, 1],[-1, 0]])
    for n in range(N):
        yo = y
        y = newton_DGM4(y,y,dt,1e-16)
    ref_q, ref_p = y[0], y[1]
    print('Reference:  q =', ref_q, '; p =', ref_p)
        
    # DGM2:
    print('DGM2:')
    errors_dg = np.array([])
    for i in range(maxit):
        N = mn*(2**i)
        dt = T/N
        y = y0
        for n in range(N):
            yo = y
            y = newton_DGM(yo,yo,dt,1e-16)
        q, p = y[0], y[1]
        print('N =', N, '; q =', q, '; p =', p)
        errors_dg = np.append(errors_dg,np.sqrt((q-ref_q)**2+(p-ref_p)**2))
    S = np.array([[0, 1],[-1, 0]])
    I = np.array([[1, 0],[0, 1]])
    print('Error factors for DGM2:', errors_dg[:-1]/errors_dg[1:])
    
    # DGM3:
    print('DGM3:')
    errors_dg3 = np.array([])
    for i in range(maxit):
        N = mn*(2**i)
        dt = T/N
        y = y0
        for n in range(N):
            yo = y
            y = newton_DGM3(yo,yo,dt,1e-16)
        q, p = y[0], y[1]
        print('N =', N, '; q =', q, '; p =', p)
        errors_dg3 = np.append(errors_dg3,np.sqrt((q-ref_q)**2+(p-ref_p)**2))
    S = np.array([[0, 1],[-1, 0]])
    I = np.array([[1, 0],[0, 1]])
    print('Error factors for DGM3:', errors_dg3[:-1]/errors_dg3[1:])
    
    # DGM4:
    print('DGM4:')
    errors_dg4 = np.array([])
    for i in range(maxit):
        N = mn*(2**i)
        dt = T/N
        y = y0
        for n in range(N):
            yo = y
            y = newton_DGM4(yo,yo,dt,1e-16)
        q, p = y[0], y[1]
        print('N =', N, '; q =', q, '; p =', p)
        errors_dg4 = np.append(errors_dg4,np.sqrt((q-ref_q)**2+(p-ref_p)**2))
    S = np.array([[0, 1],[-1, 0]])
    I = np.array([[1, 0],[0, 1]])
    print('Error factors for DGM4:', errors_dg4[:-1]/errors_dg4[1:])
    
    dt = T/(mn*2**np.arange(maxit))
    fig = plt.figure(figsize=(5,4.5))
    plt.loglog(dt, errors_dg, color=(0.8,0,0.2), linewidth=2, label='DGM2')
    plt.loglog(dt, errors_dg3, color=(0.2,0.8,.8), linewidth=2, label='DGM3')
    plt.loglog(dt, errors_dg4, color=(0,0.4,1), linewidth=2, label='DGM4')
    plt.loglog(dt, .2*dt**2, 'k--', linewidth=1)
    plt.loglog(dt, .008*dt**3, 'k--', linewidth=1)  
    plt.loglog(dt, .08*dt**4, 'k--', linewidth=1)  
    plt.xlabel("Step size", fontsize=12)
    plt.ylabel("Error at $t=1$", fontsize=12)
    plt.title("Order plot", fontsize=14)
    plt.legend()
    plt.show()
    #fig.savefig('orderplot_spring.eps', format='eps', bbox_inches='tight')