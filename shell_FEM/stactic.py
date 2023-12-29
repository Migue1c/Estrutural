import numpy as np
import scipy as sp

def Bi(s1:float, index:int, r:float, vpe) -> np.ndarray:
    phi = vpe[index, 1]
    h = vpe[index, 2]
    sen_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([[-1/h, 0, 0],
                     [(1-s1)*sen_phi/r, (1-3*s1**2+2*s1**3)*cos_phi/r, h*s1*(1-2*s1+s1**2)*cos_phi/r],
                     [0, 6*(1-2*s1)/(h**2), 2*(2-3*s1)/h],
                     [0, 6*s1*(1-s1)*sen_phi/(r*h), (-1+4*s1-3*s1**2)*sen_phi/r]])

def Bj(s1:float, index:int, r:float, vpe) -> np.ndarray:
    phi = vpe[index, 1]
    h = vpe[index, 2]
    sen_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.array([[1/h, 0, 0],
                     [s1*sen_phi/r, (s1**2)*(3-2*s1)*cos_phi/r, h*(s1**2)*(-1+s1)*cos_phi/r],
                     [0, -6*(1-2*s1)/(h**2), 2*(1-3*s1)/h],
                     [0, -6*s1*(1-s1)*sen_phi/(r*h), s1*(2-3*s1)*sen_phi/r]])

def elastM(index:int, vpe, mat) -> np.ndarray:
    E = mat[int(vpe[index, 4]), 1] # mat must have more than one material so that the array is 2D by default
    t = vpe[index, 3]
    upsilon = mat[int(vpe[index, 4]), 2]
    D = (E*t)/(1-upsilon**2)*np.array([[1,upsilon, 0, 0],[upsilon, 1, 0, 0],[0, 0, (t**2/12), upsilon*(t**2/12)],[0, 0, upsilon*(t**2/12), (t**2/12)]])
                                                                                                                                    
    return D

def transM(phi) -> np.ndarray:
    T = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
    return T

def Bmatrix(s1:float, index:int, r:float, phi:float, vpe) -> np.ndarray:
    T = transM(phi)
    return np.hstack((Bi(s1,index,r, vpe)@T,Bj(s1,index,r, vpe)@T))

def Pbar(s1:float, h:float) -> np.ndarray:
    return np.array([[1-s1, 0, 0, s1, 0, 0],
                     [0, 1-3*s1**2+2*s1**3, s1*(1-2*s1+s1**2)*h, 0, (s1**2)*(3-2*s1), (s1**2)*(s1-1)*h]])

def Pmatrix(s1:float, index:int, phi:float, vpe) -> np.ndarray:
    T = transM(phi)
    h = vpe[index, 2]
    P = Pbar(s1, h)
    #print(P, '\n')
    Pi, Pj = np.hsplit(P, 2)
    #print(Pi)
    #print(Pj)
    return np.hstack((Pi@T, Pj@T))

def Kestacked(ne:int, vpe, mat, ni:int, simpson=True) -> np.ndarray: # Incoeherent integration results
    kes = np.empty((6,6,ne), dtype=float)
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        h = vpe[i, 2]
        D = elastM(i, vpe, mat)
        if simpson:
            I = np.empty((6, 6, ni+1), dtype=float)
            for j, s1 in enumerate(np.linspace(0,1,ni+1)):
                r = ri + s1*h*np.sin(phi)
                B = Bmatrix(s1,i,r,phi, vpe)
                I[:,:,j] = B.T@D@B*(r)

            ke = 2*np.pi*h*sp.integrate.simpson(I, x=None, dx=h/ni, axis=-1)
            #print(ke, '\n')
        else:
            s1_ = lambda s: (s+1)/2
            r = lambda s: ri + s1_(s)*h*np.sin(phi)
            integrand = lambda s: Bmatrix(s1_(s),i,r(s),phi, vpe).T@D@Bmatrix(s1_(s),i,r(s),phi, vpe)*(r(s))
            ke = 2*np.pi*h*((5/9)*integrand(-np.sqrt(3/5))+(8/9)*integrand(0)+(5/9)*integrand(np.sqrt(3/5)))
            #print(ke)
        kes[:,:,i] = ke
    return kes

def k_global(ne:int, vpe, mat, ni=1200, sparse=False) -> np.ndarray:
    kes = Kestacked(ne, vpe, mat, ni)
    if sparse:
        row = []
        column = []
        data = []
        for i in range(0, ne):
            for j in range(0,6):
                for k in range(0,6):
                    row.append(3*i+j)
                    column.append(3*i+k)
                    data.append(kes[j, k, i])
        #print(row)
        #print(column)
        #print(data)
        k_globalM = sp.sparse.bsr_array((data, (row, column)), shape=(3*(ne+1), 3*(ne+1)))#.toarray()     
    else:
        k_globalM = np.zeros((3*(ne+1), 3*(ne+1)), dtype='float64')
        for i in range(0,ne):
            k_globalM[3*i:3*i+6,3*i:3*i+6] = k_globalM[3*i:3*i+6,3*i:3*i+6] + kes[:,:,i]
    return k_globalM
    #print(f'Element {i+1}')
    #for j in range(0, 3*(ne+1)):
    #    for k in range(0, 3*(ne+1)):
    #        print(f'{k_globalM[j, k]:.2}', end='   ')
    #    print()
    #print('\n\n')
    #print(k_globalM)
    #print(sparse)
 

def calculate_strains_stresses(displacements, vpe, mat):
    num_nodes = len(displacements)/3
    num_elements = len(vpe)

    strains = np.zeros((num_nodes, 4))  # Matriz para armazenar as deformações de cada nó (epsilon_s, epsilon_theta, chi_s, chi_theta)
    for i in range(num_elements):
        R = vpe[i,0]
        phi = vpe[i,1]
        B = Bmatrix(0, i, R, phi, vpe)  # Obtém a matriz B para s1 = 0 
        strains[i,:] += B @ displacements[3*i:3*i+6]  # Multiplica deslocamentos pelos valores da matriz B exceto o ultimo
    i = num_elements - 1
    h = vpe[i, 2]
    B = Bmatrix(1, i, R +h*np.sin(phi), phi, vpe)  # Obtém a matriz B para s1 = 1
    strains[i+1,:] += B @ displacements[3*i:3*i+6]  # Multiplica deslocamentos pelos valores da matriz B obtendo a ultima extensao

    forças_N = np.zeros((num_nodes, 4))
    for i in range(num_elements):
        D = elastM(i, vpe, mat)
        forças_N[i,:] += D @ (strains[i,:].T)
    forças_N[i+1,:] += D @ (strains[i+1,:].T)          

    tensoes_N = np.zeros((num_nodes, 4))
    for i in range(num_elements):
        t = vpe[i, 3]
        sigma_sd = forças_N[i,0]/t -  6*forças_N[i,2]/t**2
        sigma_sf = forças_N[i,0]/t +  6*forças_N[i,2]/t**2
        sigma_td = forças_N[i,1]/t -  6*forças_N[i,3]/t**2
        sigma_tf = forças_N[i,1]/t +  6*forças_N[i,3]/t**2
        tensoes_N[i, :] = [sigma_sd, sigma_td, sigma_sf, sigma_tf]
    i += 1
    sigma_sd = forças_N[i,0]/t - 6*forças_N[i,2]/t**2
    sigma_sf = forças_N[i,0]/t + 6*forças_N[i,2]/t**2
    sigma_td = forças_N[i,1]/t - 6*forças_N[i,3]/t**2
    sigma_tf = forças_N[i,1]/t +     6*forças_N[i,3]/t**2
    tensoes_N[i, :] = [sigma_sd, sigma_td, sigma_sf, sigma_tf]
    return strains, tensoes_N

def tensões_VM(displacements, vpe, mat):
    num_nodes = len(displacements)/3
    num_elements = len(vpe)
    tensoes_N = calculate_strains_stresses(displacements, vpe, mat)
    VM = np.zeros((num_nodes, 2))
    for i in range(num_elements):
        VM[i,0] = np.sqrt(tensoes_N[i,0]**2 -tensoes_N[i,0]*tensoes_N[i,2] + tensoes_N[i,2]**2)
        VM[i,1] = np.sqrt(tensoes_N[i,1]**2 -tensoes_N[i,1]*tensoes_N[i,3] + tensoes_N[i,3]**2)
    i = num_elements + 1
    VM[i,0] = np.sqrt(tensoes_N[i,0]**2 -tensoes_N[i,0]*tensoes_N[i,2] + tensoes_N[i,2]**2)
    VM[i,1] = np.sqrt(tensoes_N[i,1]**2 -tensoes_N[i,1]*tensoes_N[i,3] + tensoes_N[i,3]**2)
    return VM