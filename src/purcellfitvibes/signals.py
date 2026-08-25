import numpy as np

def finite_chirp(t, start, duration, f0, f1, ramp_fraction=0.06):
    """Complex linear chirp, exactly zero outside its pulse interval."""
    t=np.asarray(t); u=np.zeros_like(t,dtype=complex); q=t-start
    m=(q>=0)&(q<=duration); x=q[m]
    k=(f1-f0)/duration
    phase=2*np.pi*(f0*x+0.5*k*x*x)
    env=np.ones_like(x); tr=ramp_fraction*duration
    if tr>0:
        a=x<tr; b=x>duration-tr
        env[a]=np.sin(0.5*np.pi*x[a]/tr)**2
        env[b]=np.sin(0.5*np.pi*(duration-x[b])/tr)**2
    u[m]=env*np.exp(1j*phase)
    return u
