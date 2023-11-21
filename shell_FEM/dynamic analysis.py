import numpy as np
import scipy as sp
from modal import modal_analysis


def c_global(ne, mask, mode1:float, mode2:float, zeta1=0.08, zeta2=0.08):
    global m_globalM
    global k_globalM
    # Pressuposes that the global matrices k_globalM and m_globalM are already reduced
    global c_globalM

    fraction = 2*mode1*mode2/(mode2**2-mode1**2)
    alfa = fraction*(mode2*zeta1-mode1*zeta2)
    beta = fraction*(zeta2/mode1-zeta1/mode2)

    c_globalM = alfa*m_globalM + beta*k_globalM
    return None


def dynamic_analysis(ne, u_vct:np.ndarray, mask:np.ndarray, t_final:float, delta_t:float, mode1:float, mode2:float, zeta1=0.08, zeta2=0.08, t_decay=2.5, tau=1.0, ni=1000, sparse=False, modal_done=False):
    global m_globalM
    global k_globalM

    if not modal_done:
        modal_analysis(ne, ni, sparse)
