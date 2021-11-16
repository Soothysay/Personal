import numpy as np
from tqdm import tqdm
class ODESolver:
    '''
    Superclass of solvers
    
    '''
    
    
    def __init__(self,f):
        self.f=f
    
    def forward (self):
        '''
        
        Advances Solution by 1 time step
        

        '''
        
        raise NotImplementedError
        
    def set_initial_conditions (self, U0):
        
        if isinstance(U0, (int,float)):
            # Scalar ODE
            self.number_of_equations=1
            U0=float(U0)
        else:
            # System of Multiple Equations
            U0=np.array(U0)
            self.number_of_equations=U0.size
        self.U0=U0
    
    
    def solve(self, time_points):
        
        self.t=np.asarray(time_points)
        n=self.t.size
        
        self.u=np.zeros((n,self.number_of_equations))
        
        
        self.u[0,:]=self.U0
        
        # Integrate
        
        for i in tqdm(range(n-1),ascii=True):
            self.i=i
            self.u[i+1]=self.forward()
            
        return self.u,self.t


class ForwardEuler(ODESolver):
    
    def forward(self):
        u,f,i,t=self.u,self.f,self.i,self.t
        dt=t[i+1]-t[i]
        #print('Done')
        return u[i,:]+dt*f(u[i,:],t[i])