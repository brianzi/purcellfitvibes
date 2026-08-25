import numpy as np
from functools import lru_cache


def pair_poles(poles, max_separation=90e6):
    """Exact minimum-cost perfect matching for a small unordered pole set."""
    poles=np.asarray(poles,complex); n=len(poles)
    if n%2: raise ValueError('need an even number of poles')
    def cost(i,j):
        df=abs(poles[i].real-poles[j].real)
        wi=max(-poles[i].imag,1.0); wj=max(-poles[j].imag,1.0)
        ratio=min(wi,wj)/max(wi,wj)
        return (df/max_separation)**2 + 0.25*ratio
    @lru_cache(None)
    def solve(mask):
        if mask==0: return 0.0,()
        i=(mask & -mask).bit_length()-1; rest=mask & ~(1<<i); best=(np.inf,())
        for j in range(i+1,n):
            if rest&(1<<j):
                c,p=solve(rest&~(1<<j)); c+=cost(i,j)
                if c<best[0]: best=(c,((i,j),)+p)
        return best
    _,pairs=solve((1<<n)-1)
    return [(poles[i],poles[j]) for i,j in pairs]
