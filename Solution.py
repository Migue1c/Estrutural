import numpy as np
import math
import scipy as sc



#Function to reduce matrices
def RedMatrix(m:np.ndarray, u_DOF:np.ndarray):
    	
    m_red = m
    n = u_DOF.shape[0] - 1
    a = m.shape[1]
    
    if a == 1:
        while n >= 0 :
            if u_DOF[n,0] == 0 :
                m_red = np.delete(m_red, n, axis=0)
            n -= 1
    else:
        while n >= 0 :
            if u_DOF[n,0] == 0 :
                m_red = np.delete(m_red, n, axis=0)
                m_red = np.delete(m_red, n, axis=1)
            n -= 1
    
    return m_red



#Function to add back the zeros
def RdfMatrix(m:np.ndarray, u_DOF:np.ndarray):
    #lines only
    n = u_DOF.shape[0] - 1
    i=0
    a = m.shape[1]
    nc = np.zeros((1,a))

    while i <= n :
        if u_DOF[i,0] == 0 :
            m = np.insert(m, i, nc, axis=0 )
        i += 1
    
    return m



#Static Solution:

#Sistem of linear eqs.

def StaticSolver(k:np.ndarray, f:np.ndarray, u_DOF:np.ndarray):
    
    #Reduce stiffness matrix and force vector
    k_red = RedMatrix(k, u_DOF)
    f_red = RedMatrix(f, u_DOF)

    #Find displacement vector
    u_red = np.linalg.solve(k_red,f_red)
    
    #re-add zeros to the displacement vector
    u_rdf = RdfMatrix(u_red, u_DOF)

    return u_rdf



#Modal Solution:

#Eigenvalue problem

def ModalSolver(k:np.ndarray, m:np.ndarray, u_DOF:np.ndarray):

    global eig_vals
    global eig_vect

    #Reduce stiffness and mass matrices
    k_red = RedMatrix(k, u_DOF)
    m_red = RedMatrix(m, u_DOF)

    #Solve the eigenvalue problem
    a = np.linalg.inv(m_red) @ k_red
    eig_vals, eig_vect = np.linalg.eigh(a)


'''
é necessário adicionar zeros de volta nos vetores dos valores proprioss
'''


#Dinamic Solution:

#Newmark Method

def DinamicSolver(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, x_0:np.ndarray, x_0_d:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float):

    #Matrices to store results
    global matrix_u
    global matrix_ud 
    global matrix_ud2   

    #Reduce Matrices
    k = RedMatrix(k, u_DOF)
    m = RedMatrix(m, u_DOF)
    c = RedMatrix(c, u_DOF)
    f = RedMatrix(f, u_DOF)    

    #Store starting values:
    matrix_u = x_0
    matrix_ud = x_0_d
    x_0_d2 = np.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))
    matrix_ud2 = x_0_d2
    
    #0 for Average Acceleration Method; 1 for Linear Acceleration Method
    method = 0

    if method == 0:
        #Average Acceleration Method:
        gamma = 1/2
        beta =  1/6
    else:
        #Linear Acceleration Method:
        gamma = 1/2
        beta = 1/4
    

    while tk < t_final :
        
        #Force vector for current tk
        # f = F(t = t_k)

        #Starting value [x_d2_(0)]
        x_0_d2 = np.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))

        #Time increment:
        tk += delta_t

        #Prediction:
        x_tk1_d = x_0_d + (1 - gamma) * delta_t * x_0_d2
        x_tk1 = x_0 + delta_t * x_0_d + (0.5 - beta)*(delta_t**2) * x_0_d2

        #Equilibrium eqs.:
        s = m + (gamma * delta_t * c) + (beta * (delta_t**2) * k)
        x_tk1_d2 = np.linalg.inv(s) @ (f - (c @ x_0_d) - (k @ x_0) )

        #Correction:
        x_tk1_d = x_tk1_d + delta_t * gamma * x_tk1_d2
        x_tk1 = x_tk1 + (delta_t**2) * beta * x_tk1_d2

        #store values in matrices
        matrix_u = np.append(matrix_u, x_tk1, axis=1)
        matrix_ud = np.append(matrix_ud, x_tk1_d, axis=1)
        matrix_ud2 = np.append(matrix_ud2, x_tk1_d2, axis=1)

        #reset starting values for next iteration:
        x_0 = x_tk1
        x_0_d = x_tk1_d

    #add lines with zeros to the matrices
    matrix_u = RdfMatrix(matrix_u, u_DOF)
    matrix_ud = RdfMatrix(matrix_ud, u_DOF)
    matrix_ud2 = RdfMatrix(matrix_ud2, u_DOF)