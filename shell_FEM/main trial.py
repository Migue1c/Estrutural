import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import math as m 
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
import os

def Mesh_Properties():
 
    # Number of Points to read
    df_num_rows  = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = ['NumRowsRead'], nrows = 1)
    k            = int(df_num_rows.loc[0, 'NumRowsRead'])
 
    #Materials to read
    df_matcols  = pd.read_excel('Livro1.xlsx', sheet_name = 'Materials', usecols = [0], nrows = 1)
    m           = int(df_matcols.iloc[0, 0])
    matcols     = list(range(3, 3 + m))
 
    # Reading the Input Data / Creating DataFrames
    df_read     = ['Points','z','r','thi','Conditions','Material','Conditions1','Nn', 'Loading']
    df          = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = df_read, nrows = k)
                                                                                   
    df_info     = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = ['Discontinuity'], nrows = k)
 
    df_mat      = pd.read_excel('Livro1.xlsx', sheet_name = 'Materials', usecols = matcols, nrows = 7)
    #print(df_mat)
   
    # Matriz com as propriedades do material  
    material = np.array(df_mat.values)
    #print(material)
 
    """
    df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left')
    plt.show()
    """
 
    # Creates a DataFrame with the same columns, but with no values
    empty_df        = pd.DataFrame([[None]*len(df_read)], columns = df_read)
 
 
    #The first point can't be (0,0) due to mathematical issues
    if df.loc[0, 'r'] == 0:
        df.loc[0, 'r']  =  df.loc[0, 'r'] + 10**(-6)
 
 
    # Adding a point due to the discontinuity, regarding the geometry:
    # helps on interpolation
    i=0
    while i < (len(df_info['Discontinuity'])):
 
        if df_info.loc[i, 'Discontinuity' ] == 1:
 
            df_add1 = pd.DataFrame  ({
                'Points'        : [df.loc[i, 'Points'] + 0.0001],
                'z'             : [df.loc[i, 'z'] + 10**(-6)],
                'r'             : [df.loc[i, 'r']],
                'thi'           : [df.loc[i+1, 'thi']],
                'Conditions'    : [df.loc[i, 'Conditions1']],
                'Nn'            : [df.loc[i, 'Nn']],
                'Conditions1'   : [df.loc[i, 'Conditions1']],
                'Material'      : [df.loc[i, 'Material']],
                'Loading'       : [df.loc[i, 'Loading']]
                                    })
           
            df_add2 = pd.DataFrame  ({
                'Discontinuity' : [0]
                                    })
           
            result1 = pd.concat([df.iloc[:i+1], df_add1, df.iloc[i+1:]], ignore_index=True)
            df      = result1
 
            result2 = pd.concat([df_info.iloc[:i+1], df_add2, df_info.iloc[i+1:]], ignore_index=True)
            df_info = result2
 
            df.loc[i,'Nn'] = 0
 
        i += 1
 
    # Adding empty rows, with the number of nodes necessary to have the number of elements specified by the user
    i = 0
    while i < (len(df['Nn'])):
 
        if df.loc[i, 'Nn' ] != 0:
 
            j=0
            while j < df.loc[i, 'Nn' ]:
 
                position    = i + 1
                result      = pd.concat([df.iloc[:position], empty_df, df.iloc[position:]], ignore_index=True)
                df          = result
 
                j += 1
               
        i +=1
 
    # Complement information for the New Nodes
    i = 1
    while i < (len(df['Points'])):
 
        if pd.isna(df.loc[i, 'Points']): #If the specified value is NaN
 
            df.loc[i, 'Points']         = df.loc[i-1, 'Points'] + 0.0001
            df.loc[i, 'Conditions']     = df.loc[i-1, 'Conditions1']
            df.loc[i,'Conditions1']     = df.loc[i-1, 'Conditions1']
            df.loc[i,'Material']        = df.loc[i-1, 'Material']
        i += 1  
 
 
    # Interpolation Linear Type
    columns_interpolate     = ['z', 'r', 'thi', 'Loading']
    df[columns_interpolate] = df[columns_interpolate].interpolate(method='linear')
    df.loc[len(df)-1, 'thi'] = np.nan
 
    #print(df)
 
 
    # Matriz com as coordenadas dos pontos / Malha
    mesh = np.array(df[['z','r']].values)
    #print(mesh)
   
   
   
    # Carregamento para cada nó
    pressure_nodes = np.array(df[['Loading']].values)
    #print(pressure_nodes)
 
 
 
    # Defining each condition possible for each nodes
    condition0 = [0,0,0]
    condition1 = [1,0,0]
    condition2 = [0,1,0]
    condition3 = [0,0,1]
    condition4 = [1,1,0]
    condition5 = [1,0,1]
    condition6 = [0,1,1]
    condition7 = [1,1,1]
    conditions = [condition0, condition1, condition2, condition3, condition4, condition5, condition6, condition7]
 
    # Creating the Boundary Conditions DataFrame as empty
    Boundary_Conditions =   {   'Point'         : [],
                                'Displacement'  : [],
                                'Value'         : []
                            }
    Boundary_Conditions = pd.DataFrame(Boundary_Conditions)
 
    # Adding Rows without information on column 'Value'
    i = 0
    while i < (len(df['Points'])):
        empty_df1 = {   'Point'         : [df.loc[i, 'Points' ], df.loc[i, 'Points' ], df.loc[i, 'Points' ]],
                        'Displacement'  : ['v','w','theta'],
                        'Value'         : [np.nan, np.nan, np.nan]
                    }
        empty_df1               = pd.DataFrame(empty_df1)
        result                  = pd.concat([Boundary_Conditions, empty_df1], ignore_index=True)
        Boundary_Conditions     = result
        i +=1
 
    # Filling the Boundary Conditions with the respective values
    j=0
    for i in range(0, len(Boundary_Conditions), 3):
        condition_index = int(df.loc[j, 'Conditions'])
       
        if 0 <= condition_index < len(conditions):
            Boundary_Conditions.loc[i:i+2, 'Value'] = conditions[condition_index]
       
        j += 1
 
    #print(Boundary_Conditions)
 
 
 
    # Nome do Vetor construído com os dados da coluna "Value" : u_DOF
 
    u_DOF = np.array(Boundary_Conditions["Value"].values)
 
    u_DOF = u_DOF.reshape((-1, 1))
   
    #print(u_DOF)
 
   
 
 
 
 
    # Creation of DataFrame regarding the elements
    vpe =   {   'Node_i'    : [],
                'phi'       : [],
                'h'         : [],
                'thi'       : [],
                'mat'       : []
            }
    vpe = pd.DataFrame(vpe)
 
    # Adding the empty rows necessary
    for i in range(len(df)-1):
        add =   { 'Node_i'  : [df.loc[i, 'r']],
                'phi'       : [np.nan],
                'h'         : [np.nan],
                'thi'       : [df.loc[i, 'thi']],
                'mat'       : [df.loc[i, 'Material']]
                }
        add     = pd.DataFrame(add)
        result  = pd.concat([vpe, add], ignore_index=True)
        vpe     = result
 
 
 
    # Adding the other information
    for i in range(len(df)-1):
        vpe.loc[i, 'h'] = math.sqrt( (df.loc[i+1, 'z'] - df.loc[i, 'z'])**2 + (df.loc[i+1, 'r']-df.loc[i, 'r'])**2 )
 
        if (df.loc[i+1, 'z'] - df.loc[i, 'z']) != 0:
            vpe.loc[i, 'phi'] = math.atan( (df.loc[i+1, 'r'] - df.loc[i, 'r']) / (df.loc[i+1, 'z'] - df.loc[i, 'z']) )
        else:
            if df.loc[i+1, 'r'] > df.loc[i, 'r']:
                vpe.loc[i, 'phi'] = math.pi/2
 
            elif df.loc[i+1, 'r'] < df.loc[i, 'r']:
                vpe.loc[i, 'phi'] = -(math.pi/2)
 
    #print(vpe)
 
    vpe = np.array(vpe.values)
   
    #print("vpe:\n",vpe)
 
   
    return mesh, u_DOF, vpe, material, pressure_nodes

"""
 #Graphic of the points
df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
plt.gca().invert_yaxis()
plt.legend(loc='center left')
plt.show()
"""





#SOLUÇÃO (Modified modal solution)

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

#Modal Solution: Modified
def ModalSolver(k:np.ndarray, m:np.ndarray, u_DOF:np.ndarray):

    #Reduce stiffness and mass matrices
    k_red = RedMatrix(k, u_DOF)             #must be able to run independent analysis 
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
    eig_vals = np.reshape(eig_vals,(-1,1))
    
    i=int(len(eig_vals)-1)
    while i>=0:
        if eig_vals[i,0] <= 0:
            eig_vals = np.delete(eig_vals, i, axis=0)
            eig_vect = np.delete(eig_vect, i, axis=1)
        i -= 1  
    eig_vals = np.array(eig_vals,dtype=float)
    #print("lenght valores proprios:",len(eig_vals))
    #print("lenght vetores proprios:",np.shape(eig_vect)[1])
    #print(eig_vals)

    #re-add zeros to the eigenvectors matrix
    eig_vect = RdfMatrix(eig_vect, u_DOF)

    # Calculates the angular frequencies of each mode
    ang_freq = (np.sqrt(eig_vals))
    return ang_freq, eig_vect

#Dinamic Solution:
def DinamicSolver(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, x_0:np.ndarray, x_0_d:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float, loading, t_col, P_col, press_max):

    #Matrices to store results
    global matrix_u
    global matrix_ud 
    global matrix_ud2   

    #Starting value for the force vector
    f = Carr_t(loading, tk, t_col, P_col, press_max)

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







#PARTE ALFAGEM
#ESTÁTICA

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

    #print(sparse)
 
def calculate_strains_stresses(displacements, vpe, mat):
    num_nodes = int(len(displacements)/3)
    num_elements = len(vpe)

    strains = np.zeros((num_nodes, 4))  # Matriz para armazenar as deformações de cada nó (epsilon_s, epsilon_theta, chi_s, chi_theta)
    for i in range(num_elements):
        R = vpe[i,0]
        phi = vpe[i,1]
        B = Bmatrix(0, i, R, phi, vpe)  # Obtém a matriz B para s1 = 0 
        #print("Matriz B",B)
        strains[i,:] += B @ displacements[3*i:3*i+6,0]  # Multiplica deslocamentos pelos valores da matriz B exceto o ultimo
    i = num_elements - 1
    h = vpe[i, 2]
    B = Bmatrix(1, i, R +h*np.sin(phi), phi, vpe)  # Obtém a matriz B para s1 = 1
    strains[i+1,:] += B @ displacements[3*i:3*i+6,0]  # Multiplica deslocamentos pelos valores da matriz B obtendo a ultima extensao

    forças_N = np.zeros((num_nodes, 4))
    for i in range(num_elements):
        D = elastM(i, vpe, mat)
        forças_N[i,:] += D @ (strains[i,:].T)
    forças_N[i+1,:] += D @ (strains[i+1,:].T)          

    #tensoes_N vai ser uma matriz em que cada coluna corresponde a [sigma_sd, sigma_td, sigma_sf, sigma_tf], em que d(dentro) e f(fora) 
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

def tensões_VM(displacements, vpe, tensoes_N):      #matriz de duas colunas em que a primeira corresponde a dentro da casca e a segunda a fora da casca
    num_nodes = int(len(displacements)/3)
    num_elements = len(vpe)
    VM = np.zeros((num_nodes, 2))

    for i in range(num_elements):
        VM[i,0] = np.sqrt(tensoes_N[i,0]**2 -tensoes_N[i,0]*tensoes_N[i,2] + tensoes_N[i,2]**2)
        VM[i,1] = np.sqrt(tensoes_N[i,1]**2 -tensoes_N[i,1]*tensoes_N[i,3] + tensoes_N[i,3]**2)
    i += 1
    VM[i,0] = np.sqrt(tensoes_N[i,0]**2 -tensoes_N[i,0]*tensoes_N[i,2] + tensoes_N[i,2]**2)
    VM[i,1] = np.sqrt(tensoes_N[i,1]**2 -tensoes_N[i,1]*tensoes_N[i,3] + tensoes_N[i,3]**2)
    return VM

def FS(displacements, vpe, mat, VM, tensões_N):     #FSy - deformação plastica  FSu - rutura
    VM = tensões_VM(displacements, vpe, tensões_N)
    von_mises = np.zeros(len(VM))
    for i, row in enumerate(VM):
        von_mises[i] += np.max(row)
    ne = len(vpe)
    FSy = np.empty((ne+1))
    FSU = np.empty((ne+1))
    for i in range(ne):
        FSy[i] = mat[3 ,int(vpe[i,4])]/von_mises[i]
        FSc = 10**6
        FSt = FSc
        if np.any(tensões_N[i,:] < 0):
            FSc = mat[5, int(vpe[i,4])] / np.min(tensões_N[i,:])
            #print(min(tensões_N[i,:]))
        if np.any(tensões_N[i,:] > 0):
            FSt = mat[4 ,int(vpe[i,4])] / np.max(tensões_N[i,:])
            #print(FSt)
        if np.abs(FSt) < np.abs(FSc):
            FSU[i] = FSt
            #print(FSt)
        else:
            FSU[i] = FSc
            #print(FSc)
    return FSy, FSU
'''
0-Density [kg/m^3]
1-Elastic Modulus [GPa]
2-Poisson Ratio
3-Tensile Yield Stress [MPa]
4-Tensile Strength [MPa]
5-Compressive Strength [MPa]
6-Shear Strength [MPa]
'''








#CARREGAMENTO

def pressao(w,nev,nl,press_est,nn):
    pressure = np.zeros(nn)
    for i in range(0, nl):
        #y = np.array([df.loc[i, 'pressure'], df.loc[i+1, 'pressure']]) #same
        y = press_est
        y_axis = np.linspace(y[i],y[i+1],nev[i], endpoint=False)#y_axis = np.linspace(max(y), min(y), nev[i] + 1)#same
        pressure[w[i] - nev[i] : w[i]] += y_axis#press_node#create a two array matrix with the pressure in each node in the whole geometry
    pressure[-1] = press_est[-1]
    #print(pressure)
    return pressure

def medium_pressure(pressao, ne):
    press_medium = np.zeros(ne)
    for i in range(0, ne):
        press_medium[i] = (pressao[i+1] + pressao[i])/2
    #print(press_medium)
    return press_medium

def loading(ne: int, vpe, pressure) -> None:  # To be verified
    load_vct = np.zeros(3 * (ne + 1))
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        hi = vpe[i, 2]
        p = pressure[i] * (10**5)
        #print(phi, ri, hi, p)
        v_carr = np.zeros(6)
        A11 = 0.5 * ri * (-np.sin(phi)) - (3 / 20) * np.sin(phi) ** 2 * hi
        A12 = 0.5 * ri * np.cos(phi) - (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 3) * ri + (1 / 30) * hi * np.sin(phi))
        A14 = 0.5 * ri * (-np.sin(phi)) - (7 / 20) * hi * np.sin(phi) ** 2
        A15 = 0.5 * ri * np.cos(phi) + (7 / 20) * hi * np.sin(phi) * np.cos(phi)
        A16 = hi * (-(1 / 12) * ri - (1 / 20) * hi * np.sin(phi))
        v_carr = 2*np.pi*hi*p*np.array([A11, A12, A13, A14, A15, A16])
        #print(v_carr)
        #ef_press = np.array([0, pressure[i]])
        #s1 = lambda s: (s + 1) / 2
        #r = lambda s: ri + s1(s) * hi * np.sin(phi)
        #integrand = lambda s: ef_press.dot(Pmatrix(s1(s), i, phi)) * (r(s))
        #I = np.empty((1, 6, 200), dtype=float)
        #for j, s in enumerate(np.linspace(-1, 1, 200)):
           # I[:, :, j] = integrand(s)
        #integral = 2 * np.pi * h * sp.integrate.simpson(I, x=None, dx=h / 199, axis=-1)
        #print(integral)
        #integral = 0.347854845 * integrand(-0.861136312) + 0.652145155 * integrand(-0.339981044) + 0.652145155 * integrand(0.339981044) + 0.347854845 * integrand(0.861136312)
        load_vct[3 * i:3 * i + 6] = load_vct[3 * i:3 * i + 6] + v_carr
    #print(load for load in load_vct)
    #print(load_vct.shape)
    #print(load_vct)
    return load_vct

def Carr_t(loading,t,t_col,P_col,press_max):
    P_col = P_col * (10**5)
    p_col_adim = np.zeros(np.size(P_col))
    p_col_adim = P_col/press_max
    P_t = np.interp(t,t_col,p_col_adim)
    loading=loading*P_t
    #print(loading)
    return loading







#MODAL (modal function deprecated, Modal analysis modifiedd)

def Mestacked(ne:int, vpe, mat, ni:int, simpson=True) -> np.ndarray:
    mes = np.empty((6,6,ne), dtype=float)
    for i in range(0, ne):
        rho = mat[int(vpe[i, 4]), 0] # Specific mass for the material in the i-th element
        t = vpe[i,3]
        h = vpe[i, 2]
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        if simpson:
            I = np.empty((6, 6, ni+1), dtype=float)
            for j, s1 in enumerate(np.linspace(0,1,ni+1) ):
                r = ri + s1*h*np.sin(phi)
                P = Pmatrix(s1,i,phi, vpe)
                I[:,:,j] = (r)*P.T@P

            me = rho*t*2*sp.pi*h*sp.integrate.simpson(I, x=None, dx=h/ni, axis=-1)
        #print('The mass matrix is:\n', me)
        mes[:,:,i] = me
    return mes

def m_global(ne:int, vpe, mat, ni=1200, sparse=False) -> np.ndarray:
    global m_globalM
    mes = Mestacked(ne, vpe, mat, ni)
    if sparse:
        row = []
        column = []
        data = []
        for i in range(0, ne):
            for j in range(0,6):
                for k in range(0,6):
                    row.append(3*i+j)
                    column.append(3*i+k)
                    data.append(mes[j, k, i])
        #print(row)
        #print(column)
        #print(data)
        m_globalM = sp.sparse.bsr_array((data, (row, column)), shape=(3*(ne+1), 3*(ne+1)))#.toarray()     
    else:
        m_globalM = np.zeros((3*(ne+1), 3*(ne+1)), dtype='float64')
        for i in range(0,ne):
            m_globalM[3*i:3*i+6,3*i:3*i+6] = m_globalM[3*i:3*i+6,3*i:3*i+6] + mes[:,:,i]
    return m_globalM

def modal_analysis(vpe, u_DOF, material, k_matrix, m_matrix, ni=1200, sparse=False):
    if m_matrix == None:
        # Mass matrix calculated
        m_matrix = m_global(len(vpe), vpe, material, ni, sparse)

    # Returns eigen values in a vector and eingenvectors as columns of a matrix, negative eigenvalues and correspondent eigenvector already filtered out
    ang_freq, eig_vect = ModalSolver(k_matrix, m_matrix, u_DOF)
    #print("Frequências angulares:\n",ang_freq)
    #print("vetores proprios:\n",eig_vect)

    return ang_freq, eig_vect, m_matrix

def modal(eig_vals):
    natfreq = (np.sqrt(eig_vals)[0:2])/(2*np.pi)
    return natfreq[0], natfreq[1]
# previous function discontinued





#DINÂMICA

def c_global(k_globalM, m_globalM, mode1:float, mode2:float, zeta1=0.08, zeta2=0.08):
    # Pressuposes that the global matrices k_globalM and m_globalM are already reduced

    fraction = 2*mode1*mode2/(mode2**2-mode1**2)
    alfa = fraction*(mode2*zeta1-mode1*zeta2)
    beta = fraction*(zeta2/mode1-zeta1/mode2)

    c_globalM = alfa*m_globalM + beta*k_globalM
    return c_globalM





#Main program
def main() -> None:
    print() # program name
    print() # authors
    print('The purpose of this program is to perform either a stactic, modal or dynamic analysis of a thin shell under a pressure distribution normal to the surface.')
    print('It imports data from "Livro1.xlsx" and exports in the respective output files.\n')
    print('To chose the kind of analysis to be performed enter:')
    
    user_input = 'r'
    while True:
        if user_input == 'r':

            # File reading
            mesh, u_DOF, vpe, material, pressure_nodes = Mesh_Properties()

            # Stifness Matrix and loadding calculation
            k_matrix = k_global(len(vpe), vpe, material)
            medium_p = medium_pressure(pressure_nodes, len(vpe))
            carr = loading(len(vpe), vpe, medium_p)
            #print(carr)
            f_vect = np.reshape(carr,(-1,1))

            # Mass and damping matrix initialized or reseted in case they have been already calculated for a previous analysis with diferent input
            m_matrix = None
            c_matrix = None

        elif user_input == 's':
            
            # Displacements obtained from the linear system
            u_global = StaticSolver(k_matrix, f_vect, u_DOF)

            # Post-Processing to obtain strains, direct stresses on the inside and outsidde of the shell, von mises stresses and yiel and ultimate safety factors
            strains, tensoes_N = calculate_strains_stresses(u_global, vpe, material)
            t_VM = tensões_VM(u_global, vpe, tensoes_N)
            fsy, fsu = FS(u_global, vpe, material, t_VM, tensoes_N)

        elif user_input == 'm':
            
            # Asks how many modes the user wants by default is 6
            user_modes = int(input('\nHow many modes of vibration do you want to export? (they will be in ascending frequency ordering)'))
            if not user_modes.isinstance(int) or user_modes <=0: user_modes = 6

            if m_matrix == None:
                # Mass matrix calculated
                m_matrix = m_global(len(vpe), vpe, material, ni=1200, sparse=False)

                # Returns eigen values in a vector and eingenvectors as columns of a matrix, negative eigenvalues and correspondent eigenvector already filtered out
                ang_freq, eig_vect = ModalSolver(k_matrix, m_matrix, u_DOF)
                #print("Frequências angulares:\n",ang_freq)
                #print("vetores proprios:\n",eig_vect)



            #ang_freq = ang_freq[:user_modes]
            #eig_vect = eig_vect[:,:user_modes]
            # Call an output function

        elif user_input == 'd':
            
            user_modes = int(input('\nWhat is the second mode used to calculate the Rayleigh damping matrix? (Answer an integer greater than 1)'))

            if m_matrix == None:
                
                # Mass matrix calculated
                m_matrix = m_global(len(vpe), vpe, material, ni=1200, sparse=False)

                ang_freq, eig_vect = ModalSolver(k_matrix, m_matrix, u_DOF)

                # Calculates damping matrix
                c_matrix = c_global(k_matrix, m_matrix, ang_freq[0], ang_freq[user_modes])

            elif c_matrix == None: c_matrix = c_global(k_matrix, m_matrix, ang_freq[0], ang_freq[user_modes])



        print('s - for stactic analysis\nm - for modal analysis\nd for dynamic analysis\nr - for reading file')
        print('If you have opened the program now the file has alredu been read. If you performed changes, you can update them with "r".')

        # Receives user input
        user_input = input().lower()

        if user_input == 'e': break

    return None



mesh, u_DOF, vpe, material, pressure_nodes = Mesh_Properties()

k = k_global(len(vpe), vpe, material)
#print("matriz K \n", k)

medium_p = medium_pressure(pressure_nodes, len(vpe))
carr = loading(len(vpe), vpe, medium_p)
#print(carr)
f_vect = np.reshape(carr,(-1,1))
#print("vetor carregamento:\n",f_vect)

u_global = StaticSolver(k, f_vect, u_DOF)
#print("vetor deslocamentos:\n",u_global)

strains, tensoes_N = calculate_strains_stresses(u_global, vpe, material)
#print("strains:\n",strains)
#print("tensões:\n",tensoes_N)
t_VM = tensões_VM(u_global, vpe, tensoes_N)
#print(t_VM)

fsy, fsu = FS(u_global, vpe, material, t_VM, tensoes_N)
#print("fsy\n",fsy)
#print("fsu\n",fsu)


m = m_global(len(vpe), vpe, material, ni=1200, sparse=False)
#print(m)

eig_vals, eig_vect = ModalSolver(k, m, u_DOF)
#print("valores proprios:\n",eig_vals)
#print("vetores proprios:\n",eig_vect)

natfreq1, natfreq2 = modal(eig_vals)
#print("valores proprios:\n",natfreq1)
#print("vetores proprios:\n",natfreq2)


c = c_global(k, m, natfreq1, natfreq2)
#print(c)

#Output
#Folder Names
main_folder = "FEM Analysis - Data"
stress_folder = "Stress Data"
strain_folder = "Strain Data"
natfreqs_folder = "Natural Frequencies Data"

#Slice angle to be poltted
rev_degrees = 180

# Matrix of stress values for the surface 
#stress_matrix = np.random.rand(100, len(points_list) - 1)
stress_vect_sd= tensoes_N[:,0] 
stress_matrix_sd = np.tile(stress_vect_sd, (1, 500)).reshape(len(stress_vect_sd), -1)

with open('stress_sd.txt', 'w') as arquivo:
    for valor in stress_vect_sd:
        arquivo.write(f'{valor}\n')  # Escrever cada valor do vetor no arquivo, um por linha

# Matrix of strain values for the surface
strain_vect = strains[:,0]
strain_matrix = np.tile(strain_vect, (1, 100)).reshape(len(strain_vect), -1)

#Natural Frequencies generator (random values for example)
total_height = 1001.9
z_coordinates = np.linspace(1, 10, 10)
natural_freqs = natural_freqs = 0.03 + 0.1 * (z_coordinates/1000)**2 + 0.5 * (z_coordinates/1000)
# Writing data to a text file
with open('data_natfreqs.txt', 'w') as file:
    file.write("Z_coordinates, Natural_frequencies\n")
    for z, freq in zip(z_coordinates, natural_freqs):
        file.write(f"{z}, {freq}\n") 

#Functions

def folders_creator (main_folder,stress_folder,strain_folder,natfreqs_folder):
    # Check if the folder already exists; if not, create the folder
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
        print(f"Folder '{main_folder}' created successfully.")
    else:
        print(f"The folder '{main_folder}' already exists.")
    
    
    
    # Build the full path for the new folder inside the main folder
    stress_path = os.path.join(main_folder, stress_folder)
    strain_path = os.path.join(main_folder, strain_folder)
    natfreqs_path = os.path.join(main_folder, natfreqs_folder)

    # Check if the new folder already exists inside the main folder; if not, create the new folder
    if not os.path.exists(stress_path) : 
        os.mkdir(stress_path)
        #print(f"Folder '{stress_path}' created inside '{main_folder}' successfully.")
    #else:
        #print(f"The folder '{stress_path}' already exists inside '{main_folder}'.")
        
    # Check if the new folder already exists inside the main folder; if not, create the new folder
    if not os.path.exists(strain_path) : 
        os.mkdir(strain_path)
        #print(f"Folder '{strain_path}' created inside '{main_folder}' successfully.")
    #else:
        #print(f"The folder '{strain_path}' already exists inside '{main_folder}'.")

     # Check if the new folder already exists inside the main folder; if not, create the new folder
    if not os.path.exists(natfreqs_path) : 
        os.mkdir(natfreqs_path)
        #print(f"Folder '{natfreqs_path}' created inside '{main_folder}' successfully.")
    #else:
        #print(f"The folder '{natfreqs_path}' already exists inside '{main_folder}'.")
    
def geometry_plot(points, rev_degrees,main_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rev_angle = m.radians(rev_degrees)

    # Convert points to a numpy array
    #points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        phi = np.linspace(0, rev_angle, 100)  # Angles for complete revolution

        x1 = points[i, 1] * np.cos(phi)  # Radius
        y1 = points[i, 1] * np.sin(phi)  # Radius
        z1 = np.full_like(phi, points[i, 0])  # Height (Z) without change

        x2 = points[i + 1, 1] * np.cos(phi)  # Radius
        y2 = points[i + 1, 1] * np.sin(phi)  # Radius
        z2 = np.full_like(phi, points[i + 1, 0])  # Height (Z) without change

        # Plot the revolutions
        ax.plot(x1, y1, z1, color='blue', alpha=0.5)
        ax.plot(x2, y2, z2, color='blue', alpha=0.5)

        # Fill the space between revolutions with a surface and apply color based on stress values
        surf = ax.plot_surface(np.vstack([x1, x2]), np.vstack([y1, y2]), np.vstack([z1, z2]), color = "#14AAF5", alpha=0.8)
        #surf.set_array(stress_matrix[:, i])

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder, "geometry_plot.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def stress_plot(points, rev_degrees, stress_matrix,stress_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rev_angle = math.radians(rev_degrees)

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        phi = np.linspace(0, rev_angle, 100)  # Angles for complete revolution

        x1 = points[i, 1] * np.cos(phi)  # Radius
        y1 = points[i, 1] * np.sin(phi)  # Radius
        z1 = np.full_like(phi, points[i, 0])  # Height (Z) without change

        x2 = points[i + 1, 1] * np.cos(phi)  # Radius
        y2 = points[i + 1, 1] * np.sin(phi)  # Radius
        z2 = np.full_like(phi, points[i + 1, 0])  # Height (Z) without change

        # Plot the revolutions
        ax.plot(x1, y1, z1, color='blue', alpha=0.5)
        ax.plot(x2, y2, z2, color='blue', alpha=0.5)

        # Fill the space between revolutions with a surface and apply color based on stress values
        surf = ax.plot_surface(np.vstack([x1, x2]), np.vstack([y1, y2]), np.vstack([z1, z2]), cmap='viridis_r', alpha=0.8)
        surf.set_array(stress_matrix[:, i])

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Add a color bar with shrink to reduce its size
    cbar = fig.colorbar(surf, aspect=10, shrink=0.7, orientation='vertical', pad=0.1)
    cbar.set_label('Stress')

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,stress_folder, "stress_plot.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def strain_plot(points, rev_degrees, strain_matrix,strain_folder):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rev_angle = m.radians(rev_degrees)

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        phi = np.linspace(0, rev_angle, 100)  # Angles for complete revolution

        x1 = points[i, 1] * np.cos(phi)  # Radius
        y1 = points[i, 1] * np.sin(phi)  # Radius
        z1 = np.full_like(phi, points[i, 0])  # Height (Z) without change

        x2 = points[i + 1, 1] * np.cos(phi)  # Radius
        y2 = points[i + 1, 1] * np.sin(phi)  # Radius
        z2 = np.full_like(phi, points[i + 1, 0])  # Height (Z) without change

        # Plot the revolutions
        ax.plot(x1, y1, z1, color='blue', alpha=0.5)
        ax.plot(x2, y2, z2, color='blue', alpha=0.5)

        # Fill the space between revolutions with a surface and apply color based on strain values
        surf = ax.plot_surface(np.vstack([x1, x2]), np.vstack([y1, y2]), np.vstack([z1, z2]), cmap='plasma', alpha=0.8)
        surf.set_array(strain_matrix[:, i])  # Apply strain values to the surface

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Add a color bar with shrink to reduce its size
    cbar = fig.colorbar(surf, aspect=10, shrink=0.7, orientation='vertical', pad=0.1)
    cbar.set_label('Strain')

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,strain_folder, "strain_plot.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()
    
def plot_frequencies(coordinates_z, natural_frequencies,main_folder,natfreqs_folder):
    plt.figure()
    plt.plot(coordinates_z, natural_frequencies, marker='o', linestyle='', color='b')
    plt.xlabel('Vibration Mode')
    plt.ylabel('Natural Frequency')
    plt.title('Natural Frequencies Graph')
    plt.grid(True)

     # Adding labels with coordinates (z, natural frequency)
    for z, freq in zip(coordinates_z, natural_frequencies):
        plt.text(z,freq, f'({z:.2f},{freq:.5f})', fontsize=8, ha='right')


    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,natfreqs_folder, "natural_frequencies_graph.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    plt.show()
    plt.close()

#Plots/Figures

#Creation of folders to add files from the analysis
folders_creator(main_folder,stress_folder,strain_folder,natfreqs_folder)

#Geometry Plot
#geometry_plot(mesh,rev_degrees,main_folder)

#Stress
stress_plot(mesh,rev_degrees,stress_matrix_sd,stress_folder)

#Strain
#strain_plot(mesh,rev_degrees,strain_matrix,strain_folder)

#Natural Frequencies 
#plot_frequencies(z_coordinates,natural_freqs,main_folder,natfreqs_folder)

