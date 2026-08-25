import numpy as np
import matplotlib.pyplot as plt
from purcellfitvibes import *

truth=sorted([
 PurcellStructure(7.057e9,7.058e9,32.2e6,9.2e6),
 PurcellStructure(6.580e9,6.575e9,35.6e6,7.9e6),
 PurcellStructure(7.196e9,7.214e9,57.8e6,6.9e6),
 PurcellStructure(6.898e9,6.898e9,38.3e6,8.7e6),
 PurcellStructure(6.392e9,6.409e9,32.6e6,7.8e6)], key=lambda p:p.fr)

rng=np.random.default_rng(7); fs=2.5e9; fc=6.80e9
t=np.arange(int(1.3e-6*fs))/fs
u=finite_chirp(t,200e-9,300e-9,-550e6,550e6)
y,_=simulate_trace(t,u,fc,truth,rng=rng,delay=41e-9,noise=8e-4)

f,H=estimate_transfer(t,u,y,fc,delay=41e-9)
poles=discover_poles(f,H,len(truth))
pairs=pair_poles(poles)
pairs=sorted(pairs,key=lambda q:np.mean([q[0].real,q[1].real]))

print('broad-scan pole pairs:')
for q in pairs: print(' ',*[f'{p.real/1e9:.6f}{p.imag/1e6:+.2f}i MHz' for p in q])

fits=[]
for pair in pairs:
    g=guess_from_pair(pair); c=np.mean([z.real for z in pair])
    tt=np.arange(int(0.8e-6*fs))/fs
    uu=finite_chirp(tt,150e-9,160e-9,-70e6,70e6)
    yy,_=simulate_trace(tt,uu,c,truth,rng=rng,delay=41e-9,noise=8e-4)
    fit,delay=fit_targeted(tt,uu,yy,c,g); fits.append(fit)

print('\nrecovered structures:')
for a,b in zip(truth,fits):
    print(f'fr={a.fr/1e9:.6f} GHz -> {b.fr/1e9:.6f}; '
          f'fp={a.fp/1e9:.6f} -> {b.fp/1e9:.6f}; '
          f'kp={a.kappa_p/1e6:.2f} -> {b.kappa_p/1e6:.2f} MHz; '
          f'J={a.J/1e6:.2f} -> {b.J/1e6:.2f} MHz')

o=np.argsort(f)
plt.figure(figsize=(10,4.5))
plt.plot(f[o]/1e9,20*np.log10(np.maximum(np.abs(H[o]),1e-12)),lw=1,label='broad chirp')
for p in poles: plt.axvline(p.real/1e9,ls=':',lw=.8)
plt.xlabel('frequency (GHz)'); plt.ylabel('|transfer| (dB)')
plt.title('Broad discovery scan and fitted generic poles'); plt.grid(alpha=.25); plt.tight_layout()
plt.savefig('broad_scan.png',dpi=160)
