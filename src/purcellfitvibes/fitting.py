import numpy as np
from scipy.optimize import least_squares
from .model import PurcellStructure, s21_single


def guess_from_pair(pair, template=None):
    p=np.asarray(pair); broad=p[np.argmin(p.imag)]; narrow=p[np.argmax(p.imag)]
    if template is None:
        template=PurcellStructure(broad.real,narrow.real,30e6,8e6)
    return PurcellStructure(broad.real,narrow.real,max(5e6,-2*broad.imag),8e6,
        template.kappa_r,template.gamma_p,template.gamma_r,template.cin)


def fit_targeted(t,u,y,carrier,initial,window=70e6,nstarts=10):
    """Fit fp, fr, kappa_p, J and delay; profile complex gain/slope/feedthrough."""
    dt=np.median(np.diff(t)); fb=np.fft.fftfreq(len(t),dt); f=carrier+fb
    U=np.fft.fft(u); Y=np.fft.fft(y); center=0.5*(initial.fp+initial.fr)
    m=(np.abs(U)>0.04*np.max(np.abs(U))) & (np.abs(f-center)<window)
    ff,bb,UU,YY=f[m],fb[m],U[m],Y[m]
    scale=100e6; xn=(ff-center)/window
    def unpack(z):
        return PurcellStructure(center+scale*z[0],center+scale*z[1],
            scale*np.exp(z[2]),scale*np.exp(z[3]),initial.kappa_r,
            initial.gamma_p,initial.gamma_r,initial.cin)
    def project(z):
        h=s21_single(ff,unpack(z)); ph=np.exp(-2j*np.pi*bb*z[4]*1e-9)
        A=np.column_stack([UU*ph*h,UU*ph*h*xn,UU])
        c,*_=np.linalg.lstsq(A,YY,rcond=None)
        return A@c
    amp=max(np.median(np.abs(YY)),1e-12)
    def residual(z):
        r=(project(z)-YY)/amp; return np.r_[r.real,r.imag]
    z0=np.array([(initial.fp-center)/scale,(initial.fr-center)/scale,
                 np.log(initial.kappa_p/scale),np.log(initial.J/scale),30.0])
    lo=np.array([-0.55,-0.55,np.log(3e6/scale),np.log(1e6/scale),0.0])
    hi=np.array([ 0.55, 0.55,np.log(100e6/scale),np.log(35e6/scale),100.0])
    rng=np.random.default_rng(456); best=None
    for k in range(nstarts):
        z=z0.copy()
        if k:
            z[:2]+=rng.normal(0,0.08,2); z[2:4]+=rng.normal(0,0.35,2); z[4]+=rng.normal(0,12)
        z=np.clip(z,lo+1e-8,hi-1e-8)
        r=least_squares(residual,z,bounds=(lo,hi),loss='soft_l1',max_nfev=900)
        ss=np.sum(residual(r.x)**2)
        if best is None or ss<best[0]: best=(ss,r)
    return unpack(best[1].x), best[1].x[4]
