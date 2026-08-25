import numpy as np
from purcellfitvibes import PurcellStructure, hybrid_poles, finite_chirp, pair_poles

def test_hybrid_poles_are_stable():
    p=PurcellStructure(6.9e9,6.905e9,38e6,8e6)
    q=hybrid_poles(p)
    assert len(q)==2 and np.all(q.imag<0)

def test_finite_chirp_is_finite():
    t=np.arange(1000)*1e-9; u=finite_chirp(t,200e-9,300e-9,-50e6,50e6)
    assert np.all(u[t<200e-9]==0) and np.all(u[t>500e-9]==0)

def test_pairing():
    p=np.array([1-10j,100-8j,3-1j,102-1j],complex)
    pairs=pair_poles(p,max_separation=20)
    centers=sorted(round(np.mean([a.real,b.real])) for a,b in pairs)
    assert centers==[2,101]
