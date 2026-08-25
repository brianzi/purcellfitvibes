import numpy as np
from .model import s21_multi, gamma_input

def simulate_trace(t, u, carrier, structures, *, delay=41e-9,
                   gain_slope=0.08+0.03j, noise=0.0015,
                   echo_amplitude=0.0, echo_delay=55e-9, rng=None):
    rng=np.random.default_rng() if rng is None else rng
    dt=np.median(np.diff(t)); fb=np.fft.fftfreq(len(t),dt); f=carrier+fb
    U=np.fft.fft(u); x=(f-carrier)/max(np.ptp(fb),1.0)
    insertion=1-gamma_input(f, structures[0].cin)
    chain=insertion*np.exp(-2j*np.pi*fb*delay)*(1+gain_slope*x)
    H=chain*s21_multi(f,structures)
    if echo_amplitude:
        H*=1+echo_amplitude*np.exp(-2j*np.pi*fb*echo_delay+0.4j)
    y=np.fft.ifft(U*H)
    y += noise*(rng.normal(size=len(t))+1j*rng.normal(size=len(t)))
    return y, H
