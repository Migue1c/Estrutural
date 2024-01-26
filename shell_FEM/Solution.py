import numpy as np
import math
import scipy as sp
#from stactic import k_global
#from modal import m_global
#from dynamic import c_global
#from loading import Carr_t
#from mesh import u_DOF


#SOLUÇÃO
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
def StaticSolver(k:np.ndarray, f:np.ndarray, u_DOF:np.ndarray):
    
    #Reduce stiffness matrix and force vector
    k_red = RedMatrix(k, u_DOF)
    f_red = RedMatrix(f, u_DOF)

    #Find displacement vector
    u_red = np.linalg.solve(k_red,f_red)
    
    #re-add zeros to the displacement vector
    u_global = RdfMatrix(u_red, u_DOF)

    return u_global

#Modal Solution:
def ModalSolver(k:np.ndarray, m:np.ndarray, u_DOF:np.ndarray):

    #Reduce stiffness and mass matrices
    k_red = RedMatrix(k, u_DOF)          
    m_red = RedMatrix(m, u_DOF)

    #Solve the eigenvalue problem
    '''
    a = np.linalg.inv(m_red) @ k_red
    eig_vals, eig_vect = np.linalg.eig(a)
    print(eig_vals)
    print("vetores proprios v1:\n",eig_vect)
    '''
    eig_vals, eig_vect = sp.linalg.eig(k_red, m_red)

    #filter the results
    eig_vals = np.array(eig_vals,dtype=float)
    i=int(len(eig_vals)-1)
    while i>=0:
        if eig_vals[i] <= 0:
            eig_vals = np.delete(eig_vals, i)
            eig_vect = np.delete(eig_vect, i, axis=1)
        i -= 1  
    #print("lenght valores proprios:",len(eig_vals))
    #print("lenght vetores proprios:",np.shape(eig_vect)[1])
    #print(eig_vals)

    #re-add zeros to the eigenvectors matrix
    eig_vect = RdfMatrix(eig_vect, u_DOF)

    #sort values 
    guide_vect = np.argsort(eig_vals)
    natfreq = np.sort(np.sqrt(eig_vals))

    #sort vector
    new_mtx = np.zeros((len(eig_vect),len(guide_vect)))
    n=0
    for i in guide_vect:
        new_mtx[:,n] = eig_vect[:,i]
        n += 1
    eig_vect = new_mtx

    return natfreq, eig_vect

#Dinamic Solution:


#Original version
def DynamicSolverOG(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, x_0:np.ndarray, x_0_d:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float):

    #Matrices to store results
    global matrix_u
    global matrix_ud 
    global matrix_ud2   

    #Starting value for the force vector
    #f = Carr_t(tk)

    #Reduce Matrices
    k = RedMatrix(k, u_DOF)
    m = RedMatrix(m, u_DOF)
    c = RedMatrix(c, u_DOF)
    f = RedMatrix(f, u_DOF)

    #on the final version add an "if" to check if vectors already reduced or not
    #or something to define them as  0
    x_0 = RedMatrix(x_0, u_DOF)
    x_0_d = RedMatrix(x_0_d, u_DOF)

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
        # f = Carr_t(tk)
        # f = RedMatrix(f, u_DOF)

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


#STATIC TEST VERSION
def DynamicSolver(k:np.ndarray, m:np.ndarray, c:np.ndarray, u_DOF:np.ndarray, t_col, p_col, vpe, ne, pressure_nodes):

    #static test only
    #Reduce Matrices
    k = RedMatrix(k, u_DOF)
    m = RedMatrix(m, u_DOF)
    c = RedMatrix(c, u_DOF)

    #Define starting values vector (reduced)
    l = k.shape[0]          #sem -1 burro
    x_0 = np.zeros([l,1])
    x_0_d = np.zeros([l,1])
    x_0_d2 = np.zeros([l,1])

    #Define matrices to store results
    matrix_u = x_0
    matrix_ud = x_0_d
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
    
    #time constraints
    l = t_col.shape[0] - 1 
    tk = t_col[0,0]
    t_final = t_col[l,0]
    fg = 0
    
    while tk <= t_final :
        
        #Force vector for current tk
        f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
        f = RedMatrix(f, u_DOF)

        #Starting value [x_d2_(0)]
        x_0_d2 = np.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))
        
        #Time increment:
        tk0 = tk
        fg += 1
        tk = t_col[fg,0]
        delta_t = tk - tk0
        
        #Prediction:
        x_1_d = x_0_d + (1 - gamma) * delta_t * x_0_d2
        x_1 = x_0 + delta_t * x_0_d + (0.5 - beta)*(delta_t**2) * x_0_d2
        
        #Equilibrium eqs.:
        s = m + (gamma * delta_t * c) + (beta * (delta_t**2) * k)
        x_1_d2 = np.linalg.inv(s) @ (f - (c @ x_0_d) - (k @ x_0) )
       
        #Correction:
        x_1_d = x_1_d + delta_t * gamma * x_1_d2
        x_1 = x_1 + (delta_t**2) * beta * x_1_d2
       
        #store values in matrices
        matrix_u = np.append(matrix_u, x_1, axis=1)
        matrix_ud = np.append(matrix_ud, x_1_d, axis=1)
        matrix_ud2 = np.append(matrix_ud2, x_1_d2, axis=1)

        #reset starting values for next iteration:
        x_0 = x_1
        x_0_d = x_1_d

    #add lines with zeros to the matrices
    matrix_u = RdfMatrix(matrix_u, u_DOF)
    matrix_ud = RdfMatrix(matrix_ud, u_DOF)
    matrix_ud2 = RdfMatrix(matrix_ud2, u_DOF)

    return matrix_u, matrix_ud, matrix_ud2

#STATIC TEST VERSION
#STIFNESS FORM (NEWMARK METHOD)
def DynamicSolverV2(k:np.ndarray, m:np.ndarray, c:np.ndarray, u_DOF:np.ndarray, t_col, p_col, vpe, ne, pressure_nodes):
    #static test only
    #Reduce Matrices
    k = RedMatrix(k, u_DOF)
    m = RedMatrix(m, u_DOF)
    c = RedMatrix(c, u_DOF)

    #Define starting values vector (reduced)
    l = k.shape[0]             #sem -1 burro
    x_0 = np.zeros([l,1])
    x_0_d = np.zeros([l,1])
    x_0_d2 = np.zeros([l,1])

    #Define matrices to store results
    matrix_u = x_0
    matrix_ud = x_0_d
    matrix_ud2 = x_0_d2
   
    #0 for Average Acceleration Method; 1 for Linear Acceleration Method
    method = 0
    if method == 0:
        #Average Acceleration Method:
        gamma = 1/2
        beta =  1/4
    else:
        #Linear Acceleration Method: 
        gamma = 1/2
        beta = 1/6
    
    #time constraints
    l = t_col.shape[0] - 1
    tk = t_col[0,0]
    t_final = t_col[l,0]
    fg = 0

    #Starting acel. value
    f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
    f = RedMatrix(f, u_DOF)
    x_0_d2 = sp.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))

    while tk < t_final :
        
        #Time increment:
        tk0 = tk
        fg += 1
        tk = t_col[fg,0]
        delta_t = tk - tk0

        #Loading vector for tk
        f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
        f = RedMatrix(f, u_DOF)

        #Stifness form 
        k_SF = ( 1 / (beta*(delta_t**2))) * m  + ( gamma / (beta * delta_t)) * c + k
        f_SF = f + (( 1 / (beta*(delta_t**2))) * m  + ( gamma / (beta * delta_t)) * c ) @ x_0 + (( 1 / (beta*delta_t)) * m + ((gamma/beta)-1) * c) @ x_0_d + (((1/(2*beta))-1) * m + (delta_t/2)*((gamma/beta)-2) * c) @ x_0_d2
        x_1 = np.linalg.solve(k_SF, f_SF)

        x_1_d2 = (1/((delta_t**2)*beta))*(x_1 - x_0 - (delta_t*x_0_d) - ((delta_t**2)*(0.5-beta)*x_0_d2))
        x_1_d = x_0_d + ((1 - gamma) * delta_t * x_0_d2) + (delta_t * gamma * x_1_d2)
 
        #store values in matrices
        matrix_u = np.append(matrix_u, x_1, axis=1)
        matrix_ud = np.append(matrix_ud, x_1_d, axis=1)
        matrix_ud2 = np.append(matrix_ud2, x_1_d2, axis=1)

        #reset starting values for next iteration:
        x_0 = x_1
        x_0_d = x_1_d
        x_0_d2 = sp.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))

    #add lines with zeros to the matrices
    matrix_u = RdfMatrix(matrix_u, u_DOF)
    matrix_ud = RdfMatrix(matrix_ud, u_DOF)
    matrix_ud2 = RdfMatrix(matrix_ud2, u_DOF)

    return matrix_u, matrix_ud, matrix_ud2

#added output limiter; removed m_1_d and m_1_d2 output;
def DynamicSolverV3(k:np.ndarray, m:np.ndarray, c:np.ndarray, u_DOF:np.ndarray, t_col, p_col, vpe, ne, pressure_nodes):
    #static test only
    #Reduce Matrices
    k = RedMatrix(k, u_DOF)
    m = RedMatrix(m, u_DOF)
    c = RedMatrix(c, u_DOF)

    #Define starting values vector (reduced)
    l = k.shape[0]            
    x_0 = np.zeros([l,1])
    x_0_d = np.zeros([l,1])

    #Define matrix to store results
    matrix_u = x_0
   
    #Newmark parameters; 0 for Average Acceleration Method; 1 for Linear Acceleration Method
    method = 0
    if method == 0:
        gamma = 1/2 #Average Acceleration Method
        beta =  1/4
    else:
        gamma = 1/2 #Linear Acceleration Method
        beta = 1/6
    
    #time constraints
    n_e = t_col.shape[0]
    l = n_e - 1
    tk = t_col[0,0]
    t_final = t_col[l,0]
    fg = 0

    #Size reduction
    ##############
    n_limit = 100   #nº of time instances to be output
    ##############
    t_col_red = np.zeros([1,1])
    t_col_red[0,0] = tk
    if n_e >= n_limit:
        n_it = math.floor(n_e/n_limit)
        n_chk = n_it

    #Starting acel. value
    f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
    f = RedMatrix(f, u_DOF)
    x_0_d2 = sp.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))

    while tk < t_final :
        
        #Time increment:
        tk0 = tk
        fg += 1
        tk = t_col[fg,0]
        delta_t = tk - tk0

        #Loading vector for tk
        f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
        f = RedMatrix(f, u_DOF)

        #Stifness form 
        k_SF = ( 1 / (beta*(delta_t**2))) * m  + ( gamma / (beta * delta_t)) * c + k
        f_SF = f + (( 1 / (beta*(delta_t**2))) * m  + ( gamma / (beta * delta_t)) * c ) @ x_0 + (( 1 / (beta*delta_t)) * m + ((gamma/beta)-1) * c) @ x_0_d + (((1/(2*beta))-1) * m + (delta_t/2)*((gamma/beta)-2) * c) @ x_0_d2
        x_1 = np.linalg.solve(k_SF, f_SF)

        x_1_d2 = (1/((delta_t**2)*beta))*(x_1 - x_0 - (delta_t*x_0_d) - ((delta_t**2)*(0.5-beta)*x_0_d2))
        x_1_d = x_0_d + ((1 - gamma) * delta_t * x_0_d2) + (delta_t * gamma * x_1_d2)

        #store values in matrices
        if n_e >= n_limit:
            if fg == n_chk:
                n_chk += n_it
                matrix_u = np.append(matrix_u, x_1, axis=1)
                t_add = np.zeros([1,1])
                t_add[0,0] = tk
                t_col_red = np.append(t_col_red, t_add, axis=1)                
        else:
            matrix_u = np.append(matrix_u, x_1, axis=1)

        #reset starting values for next iteration:
        x_0 = x_1
        x_0_d = x_1_d
        x_0_d2 = sp.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))

    #add last iteration
    matrix_u = np.append(matrix_u, x_1, axis=1)
    t_add = np.zeros([1,1])
    t_add[0,0] = t_final
    t_col_red = np.append(t_col_red, t_add, axis=1)
    
    #add lines with zeros to the matrices
    matrix_u = RdfMatrix(matrix_u, u_DOF)

    return matrix_u, t_col_red