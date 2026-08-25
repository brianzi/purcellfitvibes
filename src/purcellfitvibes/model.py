from dataclasses import dataclass
import numpy as np

Z0 = 50.0

@dataclass
class PurcellStructure:
    fp: float
    fr: float
    kappa_p: float
    J: float
    kappa_r: float = 0.15e6
    gamma_p: float = 0.20e6
    gamma_r: float = 0.10e6
    cin: float = 40e-15


def gamma_input(f, cin, z0=Z0):
    w = 2*np.pi*np.asarray(f)
    return 1/(1 + 2j*w*z0*cin)


def s21_single(f, p: PurcellStructure):
    """Normalized single-structure transmission based on Eq. C11 of arXiv:1801.07904."""
    f = np.asarray(f)
    G = gamma_input(f, p.cin)
    kt = p.kappa_p*(1 + np.real(G))/2
    fpt = p.fp + p.kappa_p*np.imag(G)/4
    da, db = fpt-f, p.fr-f
    q = p.gamma_r + p.kappa_r + 2j*db
    den = 4*p.J**2 + (p.gamma_p + kt + 2j*da)*q
    return 1 - ((1+G)/(1+np.real(G))) * kt*q/den


def s21_multi(f, structures):
    """Independent side-coupled structures on a common line."""
    f = np.asarray(f)
    h = np.ones_like(f, dtype=complex)
    for p in structures:
        h += s21_single(f, p) - 1
    return h


def hybrid_poles(p: PurcellStructure):
    """Approximate constant-environment two-mode poles, in Hz."""
    a = p.fp - 0.5j*(p.kappa_p+p.gamma_p)
    b = p.fr - 0.5j*(p.kappa_r+p.gamma_r)
    return np.linalg.eigvals(np.array([[a, p.J], [p.J, b]], complex))
