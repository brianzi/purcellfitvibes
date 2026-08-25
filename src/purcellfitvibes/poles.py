import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks


def estimate_transfer(t,u,y,carrier,min_drive=0.05,delay=0.0):
    dt=np.median(np.diff(t)); fb=np.fft.fftfreq(len(t),dt); f=carrier+fb
    U=np.fft.fft(u); Y=np.fft.fft(y)
    m=np.abs(U)>min_drive*np.max(np.abs(U))
    H=Y[m]/U[m]*np.exp(2j*np.pi*fb[m]*delay)
    o=np.argsort(f[m])
    return f[m][o], H[o]


def find_feature_centers(f,H,n_features,prominence=0.15,min_spacing_hz=60e6):
    z=-20*np.log10(np.maximum(np.abs(H),1e-12))
    df=np.median(np.diff(f)); distance=max(1,int(min_spacing_hz/df))
    peaks,_=find_peaks(z,prominence=prominence,distance=distance)
    if len(peaks)<n_features:
        peaks=np.argsort(z)[-n_features:]
    peaks=peaks[np.argsort(z[peaks])[-n_features:]]
    return np.sort(f[peaks])


def fit_two_poles(f,H,center,halfwidth=65e6):
    """Generic local two-pole rational fit; no Purcell physics is assumed."""
    m=np.abs(f-center)<halfwidth; ff=f[m]; yy=H[m]
    scale=halfwidth; x=(ff-center)/scale
    z0=np.array([-0.08,np.log(0.08),0.08,np.log(0.02)])
    def poles(z):
        return np.array([center+scale*(z[0]-1j*np.exp(z[1])),
                         center+scale*(z[2]-1j*np.exp(z[3]))])
    def residual(z):
        p=poles(z)
        A=np.column_stack([1/(ff-p[0]),1/(ff-p[1]),np.ones_like(ff),x])
        c,*_=np.linalg.lstsq(A,yy,rcond=None)
        r=A@c-yy
        return np.r_[r.real,r.imag]
    best=None; rng=np.random.default_rng(123)
    for _ in range(8):
        z=z0+rng.normal(0,[0.12,0.5,0.12,0.5])
        r=least_squares(residual,z,loss='soft_l1',max_nfev=700)
        ss=np.sum(residual(r.x)**2)
        if best is None or ss<best[0]: best=(ss,r)
    return poles(best[1].x)


def discover_poles(f,H,n_structures):
    centers=find_feature_centers(f,H,n_structures)
    return np.concatenate([fit_two_poles(f,H,c) for c in centers])
