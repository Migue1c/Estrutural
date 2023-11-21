import numpy as np
import scipy as sp
from modal import modal_analysis
from stactic import k_global # it would be interesting if it would already be reduced


def c_global(ne, mask, k_globalM, m_globalM, mode1:float, mode2:float, zeta1=0.08, zeta2=0.08):
    # Pressuposes that the global matrices k_globalM and m_globalM are already reduced

    fraction = 2*mode1*mode2/(mode2**2-mode1**2)
    alfa = fraction*(mode2*zeta1-mode1*zeta2)
    beta = fraction*(zeta2/mode1-zeta1/mode2)

    c_globalM = alfa*m_globalM + beta*k_globalM
    return c_globalM


def dynamic_analysis(ne, vpe, mat, u_DOF, mask, ni=1200, sparse=False):

    zeta1=0.08
    zeta2=0.08

    print('\n')
    print('|\ | |\ | /\ |\/| | / /_')
    print('|/ | | \|/--\|  | | \  /')
    print('version 0.1.1 beta')
    print("The dynamic analysis uses some extra parameters not read from the file.")
    print("Those are 't_final' (end of simulation in seconds), 'delta_t' (step size of iteration),")
    print("'t_decay' (time instant when load decay starts to take effect) and 'tau' (time constant of the decay)\n")
    print("All numbers must be floats, the program doesn't check variable types so it might crash in 1001 wonderful different ways otherwise.\n")
    t_final = float(input('Please give t_final:\n'))
    delta_t = float(input('Please give delta_t:\n'))
    t_decay = float(input('Please give t_decay:\n'))
    tau = float(input('Please give tau:\n'))

    print('There is also an option to initialize the stifness, mass and damping matrices as sparse matrices, it is not well established yet')
    sparse = bool(input('Do you want to initialize the stifness, mass and damping matrices as sparse (PLS answer 0)'))


    k_globalM = k_global(ne, vpe, mat, ni, sparse)
    mode1, mode2, m_globalM = modal_analysis(ne, vpe, mat, ni, sparse, is_called_from_dynamic=True)
    c_globalM = c_global(ne, mask, k_globalM, m_globalM, mode1, mode2, zeta1, zeta2)

