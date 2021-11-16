import numpy as np
from ODESolver import ForwardEuler
from matplotlib import pyplot as plt

class SIR:
    def __init__(self,nu,beta,S0,I0,R0):
        
        if isinstance(nu, (int,float)):
            # If is number
            self.nu=lambda t:nu
        elif callable(nu):
            self.nu=nu
        
        if isinstance(beta, (int,float)):
            # If is number
            self.beta=lambda t:beta
        elif callable(beta):
            self.beta=beta
        
        self.initial_conditions=[S0,I0,R0]
        
        
    def __call__(self,u,t):
        
        S,I,_=u
        
        return np.asarray([-self.beta(t)*S*I,
                           -self.beta(t)*S*I - self.nu(t)*I,
                           self.nu(t)*I
                           ])
if __name__=="__main__":
    
    sir=SIR(0.1,0.0005,1500,1,0)
    print('Yes')
    solver=ForwardEuler(sir)
    solver.set_initial_conditions(sir.initial_conditions)
    
    time_steps=np.linspace(0,60,1000)
    
    u,t=solver.solve(time_steps)
    
    plt.plot(t,u[:,0],label='Susceptible')
    plt.plot(t,u[:,1],label='Infected')
    plt.plot(t,u[:,2],label='Recovered')
    plt.legend()
    plt.show()