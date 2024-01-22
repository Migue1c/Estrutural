import numpy as np
import pandas as pd
import math
import math as m 
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import os
import warnings
import time


# Ignorar o aviso específico
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated.*")
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part.*")

ti_analise = time.time()

def Mesh_Properties():

    # Number of Points to read
    df_num_rows  = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = ['NumRowsRead'], nrows = 1)
    k            = int(df_num_rows.loc[0, 'NumRowsRead'])

    #Materials to read
    df_matcols  = pd.read_excel('Livro1.xlsx', sheet_name = 'Materials', usecols = [0], nrows = 1)
    m           = int(df_matcols.iloc[0, 0])
    matcols     = list(range(3, 3 + m))
    
    # Number of lines to read, for the loading
    df_loading_read = pd.read_excel('Livro1.xlsx', sheet_name = 'Loading', usecols = ['NumRowsRead'], nrows = 1)
    k2              = int(df_loading_read.loc[0, 'NumRowsRead'])

    # Reading the Input Data / Creating DataFrames
    df_read     = ['Points','z','r','thi','Conditions','Material','Conditions1','Nn', 'Loading', 'Discontinuity'] 
    df          = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = df_read, nrows = k) 
                                                                                    
    df_info     = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = ['Discontinuity'], nrows = k)

    df_mat      = pd.read_excel('Livro1.xlsx', sheet_name = 'Materials', usecols = matcols, nrows = 7)
    
    #print(df_mat)
    
    df_loading  = ['t', 'p1']
    df_loading  = pd.read_excel('Livro1.xlsx', sheet_name = 'Loading', usecols = df_loading, nrows = k2)
    
    #print(df_loading)
    
    # Loading matrix 
    t_col = np.array(df_loading[['t']].values)  #column vector
    #print(t_col)
    
    p_col = np.array(df_loading[['p1']].values) #column vector
    #print(p_col)
    
    # Matrix with the properties of the materials
    material = np.array(df_mat.values)
    
    #print(material)
    
    #print(df)


    # Creates a DataFrame with the same columns, but with no values
    empty_df        = pd.DataFrame(np.nan, index=[0], columns = df_read)


    #print(empty_df)
    
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
                'Loading'       : [df.loc[i, 'Loading']],
                'Discontinuity' : [df.loc[i, 'Discontinuity']]
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

        if not pd.isna(df.loc[i, 'Nn']) and df.loc[i, 'Nn'] != 0:
            j=0
            while j < df.loc[i, 'Nn' ]:

                position    = i + 1
                result      = pd.concat([df.iloc[:position], empty_df, df.iloc[position:]], ignore_index=True)
                df          = result

                j += 1
                
        i +=1

    #print(df)
    
    # Complement information for the New Nodes
    i = 1
    while i < (len(df['Points'])):

        if pd.isna(df.loc[i, 'Points']): #If the specified value is NaN

            df.loc[i, 'Points']         = df.loc[i-1, 'Points'] + 0.0001
            df.loc[i, 'Conditions']     = df.loc[i-1, 'Conditions1']
            df.loc[i,'Conditions1']     = df.loc[i-1, 'Conditions1']
            df.loc[i,'Material']        = df.loc[i-1, 'Material']
            df.loc[i,'Discontinuity']        = 0
        i += 1  
    
    #print(df)
    
    
    # Creating the Mesh, with a higher density of elements close to nodes with Boundary Conditions or Discontinuity
    
        #Calcular o vetor de direção entre os nós iniciais
        #Normalizar o vetor        
        #Calcular o ponto e adicionar ao DataFrame
        
    if 1==0: ################## A alterar, para variar com os dados inseridos no excel
        i = 0
        while i < (len(df['Points'])):
            
            if not pd.isna(df.loc[i, 'Nn']) and (df.loc[i, 'Nn'] > 0):
                
                point1 = np.array(df.loc[i, ['r','z']].values)
                #print(point1)
                j = i + 1 + df.loc[i, 'Nn']
                point2 = np.array(df.loc[j, ['r','z']].values)
                #print(point2)
                
                v = point2 - point1 # Direction Vector 
                u = v / np.linalg.norm(v) # Normalize the direction vector
                distance = np.linalg.norm(v) #Distance between points
            
                
                # If both points need more elements closer to them
                if ( df.loc[i, 'Discontinuity'] == 1 or df.loc[i, 'Conditions'] != 7 ) and ( df.loc[j, 'Discontinuity'] == 1 or df.loc[j, 'Conditions'] != 7 ):
                    
                    
                    k = df.loc[i, 'Nn']
                    q = i
                    while k >= 1:
                        
                        add = (distance / 2) + ( math.cos( (math.pi / (df.loc[i, 'Nn'] + 1) )* k )) * (distance / 2) # distance to add from point1
                        point3 = point1 + u*add
                        df.loc[q+1, 'r'] = point3[0]
                        df.loc[q+1, 'z'] = point3[1]
                        
                        k = k - 1
                        q = q + 1

                # If point 1 needs more elements closer
                if ( df.loc[i, 'Discontinuity'] == 1 or df.loc[i, 'Conditions'] != 7 ) and ( df.loc[j, 'Discontinuity'] == 0 and df.loc[j, 'Conditions'] == 7 ):
                    
                    k = 1
                    q = i
                    while k <= df.loc[i, 'Nn']:
                        
                        add = ( math.cos( (((math.pi)/2) / (df.loc[i, 'Nn'] + 1) )* k )) * distance 
                        point3 = point2 - u*add
                        df.loc[q+1, 'r'] = point3[0]
                        df.loc[q+1, 'z'] = point3[1]
                        
                        k = k + 1
                        q = q + 1
                
                # If point2 needs more elements closer
                if ( df.loc[i, 'Discontinuity'] == 0 and df.loc[i, 'Conditions'] == 7 ) and ( df.loc[j, 'Discontinuity'] == 1 or df.loc[j, 'Conditions'] != 7 ):
                
                    k = df.loc[i, 'Nn']
                    q = i
                    while k >= 1:
                        
                        add = ( math.cos( (((math.pi)/2) / (df.loc[i, 'Nn'] + 1) )* k )) * distance 
                        point3 = point1 + u*add
                        df.loc[q+1, 'r'] = point3[0]
                        df.loc[q+1, 'z'] = point3[1]
                        
                        k = k - 1
                        q = q + 1
                        
                #If neither points need more elements, we add elements with the same length
                if ( df.loc[i, 'Discontinuity'] == 0 and df.loc[i, 'Conditions'] == 7 ) and ( df.loc[j, 'Discontinuity'] == 0 and df.loc[j, 'Conditions'] == 7 ):
                    
                    columns_interpolate     = ['z', 'r']
                    df[columns_interpolate] = df[columns_interpolate].interpolate(method='linear')

            i = i + 1
        
           
    # Interpolation Linear Type
    columns_interpolate     = ['thi', 'Loading']
    df[columns_interpolate] = df[columns_interpolate].interpolate(method='linear')
    df.loc[len(df)-1, 'thi'] = np.nan 
    
    if 1==1: ################## A alterar, para variar com os dados inseridos no excel
        # Interpolation Linear Type
        columns_interpolate     = ['thi', 'Loading', 'z', 'r']
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

    '''
    #Graphic of the points
    df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left')
    plt.show()
    '''
    
    #print(material)
    #print(vpe)
    
    return mesh, u_DOF, vpe, material, pressure_nodes, t_col, p_col

"""
#Graphic of the points
df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
plt.gca().invert_yaxis()
plt.legend(loc='center left')
plt.show()
"""

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
    E = mat[1, int(vpe[index, 4])-1] # mat must have more than one material so that the array is 2D by default
    t = vpe[index, 3]
    upsilon = mat[2, int(vpe[index, 4])-1]
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

            ke = 2*np.pi*h*sp.integrate.simpson(I, x=None, dx=1/ni, axis=-1)
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
        FSy[i] = mat[3 ,int(vpe[i,4])-1]/von_mises[i]    
        FSc = 10**6
        FSt = FSc
        if np.any(tensões_N[i,:] < 0):
            FSc = mat[5, int(vpe[i,4])-1] / np.min(tensões_N[i,:])
            #print(min(tensões_N[i,:]))
        if np.any(tensões_N[i,:] > 0):
            FSt = mat[4 ,int(vpe[i,4])-1] / np.max(tensões_N[i,:])
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

def medium_pressure(pressao, ne):
    pressao = pressao.reshape(-1)
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
        A12 = 0.5 * ri * np.cos(phi) + (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 12) * ri + (1 / 30) * hi * np.sin(phi))
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

def Carr_t(loading,t,t_col,P_col):
    P_col = P_col * (10**5)
    press_max = np.amax(P_col)
    p_col_adim = np.zeros(np.size(P_col))
    p_col_adim = P_col/press_max
    P_t = np.interp(t,t_col,p_col_adim)
    loading=loading*P_t
    #print(loading)
    return loading

#MODAL

def Mestacked(ne:int, vpe, mat, ni:int, simpson=True) -> np.ndarray:
    mes = np.empty((6,6,ne), dtype=float)
    for i in range(0, ne):
        rho = mat[int(vpe[i, 4])-1, 0] # Specific mass for the material in the i-th element
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

            me = rho*t*2*sp.pi*h*sp.integrate.simpson(I, x=None, dx=1/ni, axis=-1)
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

#def modal_analysis(ne, vpe, u_DOF, mat, ni=1200, sparse=False, is_called_from_dynamic=False):
    k_M = k_global(ne, vpe, mat, ni, sparse)
    m_globalM = m_global(ne, vpe, mat, ni, sparse)
    if is_called_from_dynamic:
        natfreq = (np.sqrt(sp.linalg.eigh(k_globalM, m_globalM, eigvals_only=True))[0:2])/(2*np.pi)
        return natfreq[0], natfreq[1], m_globalM, k_globalM
    else:
        ModalSolver(k_global,m_global, u_DOF)
    output = np.array[eig_vals,eig_vect]

def modal(eig_vals):
    natfreq = (np.sqrt(eig_vals)[0:2])/(2*np.pi)
    return natfreq[0], natfreq[1]

#DINÂMICA

def c_global(k_globalM, m_globalM, mode1:float, mode2:float, zeta1=0.08, zeta2=0.08):
    # Pressuposes that the global matrices k_globalM and m_globalM are already reduced

    fraction = 2*mode1*mode2/(mode2**2-mode1**2)
    alfa = fraction*(mode2*zeta1-mode1*zeta2)
    beta = fraction*(zeta2/mode1-zeta1/mode2)

    c_globalM = alfa*m_globalM + beta*k_globalM
    return c_globalM

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

    return eig_vals, eig_vect

#Dinamic Solution:
def DinamicSolver(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, x_0:np.ndarray, x_0_d:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float, loading, t_col, P_col):

    #Matrices to store results
    global matrix_u
    global matrix_ud 
    global matrix_ud2   

    #Starting value for the force vector
    f = Carr_t(loading, tk, t_col, P_col)

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
        f = Carr_t(loading, tk, t_col, P_col)
        f = RedMatrix(f, u_DOF)

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


#############################################################################################################################################
#############################################################################################################################################
#############################################################################################################################################


#LEITURA DO FICHEIRO
mesh, u_DOF, vpe, material, pressure_nodes, t_col, P_col = Mesh_Properties()

#ANÁLISE ESTÁTICAs
#MATRIZ K
k = k_global(len(vpe), vpe, material)                       #calculo matriz K
#k_df = pd.DataFrame(k)                                      #converter pra dataframe
#k_df.to_excel('k.xlsx', index=False)                        #guardar DF no excel

#CARREGAMENTO
medium_p = medium_pressure(pressure_nodes, len(vpe))        #calcular pressão média
carr = loading(len(vpe), vpe, medium_p)                     #calcular vetor de carregamento (como array 1D)
f_vect = np.reshape(carr,(-1,1))                            #converter carr para um vetor (array 2D)
#print("vetor carregamento:\n",f_vect)                   

#SOLUÇÃO E POS-PROCESSAMENTO ESTÁTICA
u_global = StaticSolver(k, f_vect, u_DOF)                   #calculo dos deslocamentos
#print("vetor deslocamentos:\n",u_global)               
strains, tensoes_N = calculate_strains_stresses(u_global, vpe, material)    #calculo das extensões e tensões diretas (e_s, e_th, x_s, x_th)
t_VM = tensões_VM(u_global, vpe, tensoes_N)                 #calculo das tensões de von-misses (t_s_d, t_th_d, t_s_f, t_th_f)
fsy, fsu = FS(u_global, vpe, material, t_VM, tensoes_N)     #calculo dos fatores de segurança (fsy-cedencia, fsu-rutura)
#print("strains:\n",strains)
#print("tensões:\n",tensoes_N)
#print("t_VM:\n",t_VM)
#print("fsy:\n",fsy)
#print("fsu:\n",fsu)

#ANÁLISE MODAL
#MATRIZ M
m_gl = m_global(len(vpe), vpe, material, ni=1200, sparse=False)#calculo matriz M
#m_df = pd.DataFrame(m)                                      #converter pra dataframe
#m_df.to_excel('m.xlsx', index=False)                        #guardar DF no excel
#print(m)

#SOLUÇÃO E POS-PROCESSAMENTO MODAL
eig_vals, eig_vect = ModalSolver(k, m_gl, u_DOF)               #calculo valores e vetores próprios
natfreq1, natfreq2 = modal(eig_vals)                        #calculo das frequências naturais para amortecimento
#print("valores proprios:\n",eig_vals)                      
#print("vetores proprios:\n",eig_vect)                   
#print("freq. natural 1:\n",natfreq1)
#print("freq. natural 2:\n",natfreq2)


#ANÁLISE DINÂMICA
#MATRIZ C
c = c_global(k, m_gl, natfreq1, natfreq2)                      #calculo matriz C
c_df = pd.DataFrame(c)                                      #converter pra dataframe
c_df.to_excel('c.xlsx', index=False)                        #guardar DF no excel
#print(c)

#DinamicSolver(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, x_0:np.ndarray, x_0_d:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float, loading, t_col, P_col)









tf_analise = time.time()

#############################################################################################################################################

#Output
print()
print("Output -> On")
print()

ti_output = time.time()

#Slice angle to be poltted
rev_degrees = 180
rev_points = 250 

#Folder Names
main_folder = "FEM Analysis - Data"
stress_folder = "Stress Data"
strain_folder = "Strain Data"
displacement_folder = 'Displacement Data'
natfreqs_folder = "Natural Frequencies Data"
safety_folder = 'Safety Factor Data'

static_folder = 'Static Analysis'
modal_folder = 'Modal Analysis'
dynamic_folder = 'Dynamic Analysis'

#Geometry - Output
geometry_photo = "Geometry.png"

#STRESS DATA

#Stress_SD - Output
stress_sd_photo = "Stress-SD.png"   #Photo Name
stress_sd_graph = "Stress-SD"       #Graph Title
stress_sd_file = "Stress-SD.txt"    #File Name
# Matrix of sigma_sd
stress_vect_sd= tensoes_N[:,0]
stress_matrix_sd = np.tile(stress_vect_sd, (1, rev_points)).reshape(len(stress_vect_sd), -1)

#Stress_SD - Output
stress_td_photo = "Stress-TD.png"   #Photo Name
stress_td_graph = "Stress-TD"       #Graph Title
stress_td_file = "Stress-TD.txt"    #File Name
# Matrix of sigma_td (Verificar que métrica é!!!) 
stress_vect_td= tensoes_N[:,1]
stress_matrix_td = np.tile(stress_vect_td, (1, rev_points)).reshape(len(stress_vect_td), -1)

#Stress_SF - Output
stress_sf_photo = "Stress-SF.png"   #Photo Name
stress_sf_graph = "Stress-SF"       #Graph Title
stress_sf_file = "Stress-SF.txt"    #File Name
# Matrix of sigma_td (Verificar que métrica é!!!) 
stress_vect_sf= tensoes_N[:,2]
stress_matrix_sf = np.tile(stress_vect_sf, (1, rev_points)).reshape(len(stress_vect_sf), -1)

#Stress_TF - Output
stress_tf_photo = "Stress-TF.png"   #Photo Name
stress_tf_graph = "Stress-TF"       #Graph Title
stress_tf_file = "Stress-TF.txt"    #File Name
# Matrix of sigma_td (Verificar que métrica é!!!) 
stress_vect_tf= tensoes_N[:,3]
stress_matrix_tf = np.tile(stress_vect_tf, (1, rev_points)).reshape(len(stress_vect_tf), -1)

#Stress Von Mises Inside - Output
stress_vm_inside_photo = "Stress VM - Inside.png"
stress_vm_inside_graph = "Stress VM - Inside"
stress_vm_inside_file =  "Stress VM - Inside.txt"
stress_vect_vm_inside = t_VM[:,0]
stress_matrix_vm_inside = np.tile(stress_vect_vm_inside, (1, rev_points)).reshape(len(stress_vect_vm_inside), -1)

#Stress Von Mises Inside - Output
stress_vm_outside_photo = "Stress VM - Outside.png"
stress_vm_outside_graph = "Stress VM - Outside"
stress_vm_outside_file =  "Stress VM - Outside.txt"
stress_vect_vm_outisde = t_VM[:,1]
stress_matrix_vm_outside = np.tile(stress_vect_vm_outisde, (1, rev_points)).reshape(len(stress_vect_vm_outisde), -1)

#Displacement

#Displacement V - Output
displacement_v_photo = "Displacement-V.png"
displacement_v_graph = "Displacement-V"
displacement_v_file = "Displacement-V.txt"
displacement_vect_v = u_global[0::3,0]
displacement_matrix_v = np.tile(displacement_vect_v, (1, rev_points)).reshape(len(displacement_vect_v), -1)

#Displacement W - Output
displacement_w_photo = "Displacement-W.png"
displacement_w_graph = "Displacement-W"
displacement_w_file = "Displacement-W.txt"
displacement_vect_w = u_global[1::3,0]
displacement_matrix_w = np.tile(displacement_vect_w, (1, rev_points)).reshape(len(displacement_vect_w), -1)

#Displacement W - Output
displacement_theta_photo = "Displacement-Theta.png"
displacement_theta_graph = "Displacement-Theta"
displacement_theta_file = "Displacement-Theta.txt"
displacement_vect_theta = u_global[2::3,0]
displacement_matrix_theta = np.tile(displacement_vect_theta, (1, rev_points)).reshape(len(displacement_vect_theta), -1)

#STRAIN DATA

#Strain: Epsilon_S - Output
strain_epsilon_s_photo = "Strain-Epsilon S.png"
strain_epsilon_s_graph = "Strain-Epsilon S"
strain_epsilon_s_file = "Strain-Epsilon S.txt"
strain_vect_epsilon_s= strains[:,0]
stress_matrix_epsilon_s = np.tile(strain_vect_epsilon_s, (1, rev_points)).reshape(len(strain_vect_epsilon_s), -1)

#Strain: Epsilon_Theta - Output
strain_epsilon_theta_photo = "Strain-Epsilon Theta.png"
strain_epsilon_theta_graph = "Strain-Epsilon Theta"
strain_epsilon_theta_file = "Strain-Epsilon Theta.txt"
strain_vect_epsilon_theta= strains[:,1]
stress_matrix_epsilon_theta = np.tile(strain_vect_epsilon_theta, (1, rev_points)).reshape(len(strain_vect_epsilon_theta), -1)

#Strain: Chi_S - Output
strain_chi_s_photo = "Strain-Chi S.png"
strain_chi_s_graph = "Strain-Chi S"
strain_chi_s_file = "Strain-Chi S.txt"
strain_vect_chi_s= strains[:,2]
stress_matrix_chi_s = np.tile(strain_vect_chi_s, (1, rev_points)).reshape(len(strain_vect_chi_s), -1)

#Strain: Chi_Theta - Output
strain_chi_theta_photo = "Strain-Chi Theta.png"
strain_chi_theta_graph = "Strain-Chi Theta"
strain_chi_theta_file = "Strain-Chi Theta.txt"
strain_vect_chi_theta= strains[:,3]
stress_matrix_chi_theta = np.tile(strain_vect_chi_theta, (1, rev_points)).reshape(len(strain_vect_chi_theta), -1)

#FSy - deformação plastica  FSu - rutura

#Safety Factor: Fsy - Output
safety_fsy_photo = "Safety Factor - FSy.png"
safety_fsy_graph = "Safety Factor - FSy"
safety_fsy_file = "Safety Factor FSy.txt"
safety_vect_fsy= fsy
safety_matrix_fsy = np.tile(safety_vect_fsy, (1, rev_points)).reshape(len(safety_vect_fsy), -1)

#Safety Factor: Fsy - Output
safety_fsu_photo = "Safety Factor - FSu.png"
safety_fsu_graph = "Safety Factor - FSu"
safety_fsu_file = "Safety Factor FSu.txt"
safety_vect_fsu= fsu
safety_matrix_fsu = np.tile(safety_vect_fsu, (1, rev_points)).reshape(len(safety_vect_fsu), -1)

#Natural Frequencies
natfreqs_file = "Natural Frequencies.txt"

show = 0

#Functions
def folders_creator (main_folder,stress_folder,strain_folder,natfreqs_folder,static_folder,modal_folder,dynamic_folder,displacement_folder,safety_folder):
    # Check if the folder already exists; if not, create the folder
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
        print(f"Folder '{main_folder}' created successfully.")
    else:
        print(f"The folder '{main_folder}' already exists.")

    static_path = os.path.join(main_folder, static_folder)
    if not os.path.exists (static_path):
        os.mkdir(os.path.join(main_folder, static_folder))
        print(f"Folder '{static_folder}' created successfully.")
    else:
        print(f"The folder '{static_folder}' already exists.")

    modal_path = os.path.join(main_folder, modal_folder)
    if not os.path.exists (modal_path):
        os.mkdir(os.path.join(main_folder, modal_folder))
        print(f"Folder '{modal_folder}' created successfully.")
    else:
        print(f"The folder '{modal_folder}' already exists.")
    
    dynamic_path = os.path.join(main_folder, dynamic_folder)
    if not os.path.exists (dynamic_path):
        os.mkdir(os.path.join(main_folder, dynamic_folder))
        print(f"Folder '{dynamic_folder}' created successfully.")
    else:
        print(f"The folder '{dynamic_folder}' already exists.")
    

    # Build the full path for the new folder inside the main folder
    stress_path = os.path.join(main_folder,static_folder,stress_folder)
    displacement_path = os.path.join(main_folder,static_folder, displacement_folder)
    strain_path = os.path.join(main_folder,static_folder, strain_folder)
    natfreqs_path = os.path.join(main_folder,modal_folder, natfreqs_folder)
    safety_path = os.path.join(main_folder, static_folder, safety_folder)

    if not os.path.exists(stress_path) : 
        os.mkdir(stress_path)

    if not os.path.exists(displacement_path) : 
        os.mkdir(displacement_path)
    
    if not os.path.exists(natfreqs_path) : 
        os.mkdir(natfreqs_path)
    
    if not os.path.exists(safety_path) : 
        os.mkdir(safety_path)

    if not os.path.exists(strain_path) : 
        os.mkdir(strain_path)    
    
def geometry_plot(points, rev_degrees,main_folder,graph_name,show):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    rev_angle = math.radians(rev_degrees)

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
        ax.plot(x1, y1, z1, color='black', alpha=0.5)
        ax.plot(x2, y2, z2, color='black', alpha=0.5)

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
    plot_path = os.path.join(main_folder, graph_name)
    # Save the plot inside the folder
    plt.savefig(plot_path)
    if show == 1:
        plt.show()
    
    plt.close()
    
def graphs (points, rev_degress, metric_matrix, metric_name, metric_folder, graph_name, show, analysis_folder): #Graph generator and saver
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rev_angle = math.radians(rev_degrees)

    #cmap_eng = mcolors.LinearSegmentedColormap.from_list("custom", ["blue", "green", "yellow", "orange", "red"])

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    min_metric_value = np.min(metric_matrix)
    max_metric_value = np.max(metric_matrix)

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
        ax.plot(x1, y1, z1, color='black', alpha=1)
        ax.plot(x2, y2, z2, color='black', alpha=1)

        # Fill the space between revolutions with a surface and apply color based on stress values
        surf = ax.plot_surface(np.vstack([x1, x2]), np.vstack([y1, y2]), np.vstack([z1, z2]), cmap = 'viridis', alpha=0.80)
        surf.set_array(metric_matrix[:, i])

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Add a color bar with shrink to reduce its size
    cbar = fig.colorbar(surf, aspect=10, shrink=0.7, orientation='vertical', pad=0.1)
    cbar.set_label(graph_name)

    analysis_folder = 'Static Analysis'

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,analysis_folder,metric_folder, metric_name)
    # Save the plot inside the folder
    plt.savefig(plot_path)
    if show == 1:
        plt.show()
    plt.close()

def graphs_geometry (points, rev_degress,rev_points, main_folder, analysis_folder): #Graph generator and saver
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    rev_angle = math.radians(rev_degrees)

    #cmap_eng = mcolors.LinearSegmentedColormap.from_list("custom", ["blue", "green", "yellow", "orange", "red"])

    # Convert points to a numpy array
    points = np.array(points)

    # Sort points by height (Z) from smallest to largest
    points = points[points[:, 0].argsort()]

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    points_matrix = np.empty((len(points), 100, 3))

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        phi = np.linspace(0, rev_angle, 100)  # Angles for complete revolution

        x1 = points[i, 1] * np.cos(phi)  # Radius
        y1 = points[i, 1] * np.sin(phi)  # Radius
        z1 = np.full_like(phi, points[i, 0])  # Height (Z) without change

        x2 = points[i + 1, 1] * np.cos(phi)  # Radius
        y2 = points[i + 1, 1] * np.sin(phi)  # Radius
        z2 = np.full_like(phi, points[i + 1, 0])  # Height (Z) without change

        # Store the points in the three-dimensional matrix
        points_matrix[i, :, 0] = x1
        points_matrix[i, :, 1] = y1
        points_matrix[i, :, 2] = z1

        points_matrix[i + 1, :, 0] = x2
        points_matrix[i + 1, :, 1] = y2
        points_matrix[i + 1, :, 2] = z2

        # Plot the revolutions
        ax.plot(x1, y1, z1, color='black', alpha=1)
        ax.plot(x2, y2, z2, color='black', alpha=1)
    
    ax.invert_zaxis()

    plot_path = os.path.join(main_folder,analysis_folder, 'mesh.png')
    plt.savefig(plot_path)
    plt.close()  # Close the figure to release resources

    return points_matrix

def graphs_1(points,points_matrix,metric_matrix, metric_name, metric_folder, graph_name, show, analysis_folder, loaded_image):

    # Create a new figure
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})  # Ensure that ax is a 3D subplot

    # Find maximum height and maximum radius
    max_height = np.max(points[:, 0])  # Maximum height
    max_radius = np.max(points[:, 1])  # Maximum radius

    # Surface between revolutions of each pair of points
    for i in range(len(points) - 1):
        # Fill the space between revolutions with a surface and apply color based on stress values
        surf = ax.plot_surface(np.vstack([points_matrix[i, :, 0] , points_matrix[i + 1, :, 0] ]), 
                               np.vstack([points_matrix[i, :, 1], points_matrix[i + 1, :, 1]]), 
                               np.vstack([points_matrix[i, :, 2], points_matrix[i + 1, :, 2]]), cmap = 'viridis', alpha=0.80)
        surf.set_array(metric_matrix[:, i])

        # Plot the revolutions
        ax.plot(points_matrix[i, :, 0], points_matrix[i, :, 1], points_matrix[i, :, 2], color='black', alpha=1)
        ax.plot(points_matrix[i + 1, :, 0], points_matrix[i + 1, :, 1], points_matrix[i + 1, :, 2], color='black', alpha=1)

    ax.set_xlabel('R[mm]')
    ax.set_ylabel('')
    ax.set_zlabel('Z[mm]')

    # Set plot limits for better visualization
    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(max_height, 0)  # Set upper limit as the maximum height

    # Add a color bar using all instances of surf
    cbar = fig.colorbar(surf, aspect=10, shrink=0.7, orientation='vertical', pad=0.1)
    cbar.set_label(graph_name)

    analysis_folder = 'Static Analysis'

    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,analysis_folder,metric_folder, metric_name)
    # Save the plot inside the folder
    plt.savefig(plot_path)
    if show == 1:
        plt.show()
    plt.close() 

def files (file_name, metric_vect, main_folder, metric_folder,analysis_folder,points): #Save Array data
    #Full path to save the file inside the folder
    file_path = os.path.join(main_folder,analysis_folder,metric_folder, file_name)
    #Write file inside the folder
    with open(file_path, 'w') as file:
        for point, metric_value in zip(points[:,0], metric_vect):
            formatted_point = "{:.3f}".format(point)  # Formata o ponto para ter no máximo 3 casas decimais
            file.write(f'{formatted_point} {metric_value}\n')

def tecplot_exporter(file_name, divisions, mesh, u_global, strains, tensoes_N, t_VM, fsy, fsu):
    
    main_folder = "FEM Analysis - Data"
    static_folder = 'Static Analysis'

    rev_angle = 2 * np.pi
    n_nodes = len(mesh)
    angles = np.linspace(0, rev_angle , divisions, endpoint=True)

    x = np.empty([n_nodes, divisions])
    y = np.empty([n_nodes, divisions])
    for i in range(0, n_nodes):
        x[i, :] = mesh[i,1]*np.cos(angles)
        y[i, :] = mesh[i,1]*np.sin(angles)

    file_path = os.path.join(main_folder, static_folder, file_name)

    #Write file inside the folder
    with open(file_path, 'w') as file:
        file.write("TITLE = \"Shell\"\n")
        file.write("VARIABLES = x, y, z, v, w, theta, es, et, xs, xt, ssd, std, ssf, stf, VM_d, VM_f, fsy, fsu\n")
        file.write(f"ZONE T=\"undeformed\", I={n_nodes:04d} J={divisions:04d}\n")

        for i in range(0, divisions):
            for j in range(0, n_nodes):
                #print(f'{x[j,i]}  {y[j,i]}  {mesh[j,0]}  {strains[j,1]}')
                file.write(f'{x[j,i]:.7e}  {y[j,i]:.7e}  {mesh[j,0]:.7e}  {u_global[3*j,0]:.7e}  {u_global[3*j+1,0]:.7e}  {u_global[3*j+2,0]:.7e}  {strains[j,0]:.7e}  {strains[j,1]:.7e}  {strains[j,2]:.7e}  {strains[j,3]:.7e}  {tensoes_N[j,0]:.7e}  {tensoes_N[j,1]:.7e}  {tensoes_N[j,2]:.7e}  {tensoes_N[j,3]:.7e}  {t_VM[j,0]:.7e}  {t_VM[j,1]:.7e}  {fsy[j]:.7e}  {fsu[j]:.7e}\n')

def nat_freqs(natural_frequencies,main_folder,metric_folder,file_name,show, analysis_folder):
    
    modes_number = len(natural_frequencies)  # Set the size of the sequence
    modes = [i + 1 for i in range(modes_number)]

    file_path = os.path.join(main_folder,analysis_folder, metric_folder, file_name)
    # Writing data to a text file
    with open(file_path, 'w') as file:
        file.write("Natural_frequencies\n")
        for z, freq in zip(modes, eig_vals):
            file.write(f"{freq}\n") 
    
    plt.figure()
    plt.plot(modes, natural_frequencies, marker='o', markersize="5" ,linestyle='', color='b')
    plt.xlabel('Vibration Mode')
    plt.ylabel('Natural Frequency')
    plt.title('Natural Frequencies Graph')
    plt.grid(True)

    # Adding labels with coordinates (modes, natural frequency)
    for mode, freq in zip(modes, natural_frequencies):
        plt.text(mode, float(freq), f'{float(freq):.2f}', fontsize=8, ha='center', va='bottom', color='black')


    # Full path to save the plot inside the folder
    plot_path = os.path.join(main_folder,analysis_folder,natfreqs_folder, "natural_frequencies_graph.png")
    # Save the plot inside the folder
    plt.savefig(plot_path)
    if show ==1:
        plt.show()
    plt.close()

###################################################################################

#Plots/Figures

#Generic Geometry
points_matrix = graphs_geometry (mesh, rev_degrees,rev_points, main_folder, static_folder)
image_path = os.path.join(main_folder,static_folder,'mesh.png')
# Load the existing image to be colored
loaded_image = plt.imread(image_path)

#Creation of folders to add files from the analysis (Mandatory)
folders_creator(main_folder,stress_folder,strain_folder,natfreqs_folder,static_folder,modal_folder,dynamic_folder,displacement_folder,safety_folder)

#Geometry Plot
geometry_plot(mesh,rev_degrees,main_folder,geometry_photo, show)

#Tecplot data exporter for 3D
tecplot_exporter('output_export_tecplot_3d.txt', rev_points, mesh, u_global, strains, tensoes_N, t_VM, fsy, fsu)



#Graphs

#Stress - Graphs
graphs (mesh, rev_degrees, stress_matrix_sd, stress_sd_photo, stress_folder, stress_sd_graph,show,static_folder)

graphs (mesh, rev_degrees, stress_matrix_td, stress_td_photo, stress_folder, stress_td_graph,show,static_folder)
graphs (mesh, rev_degrees, stress_matrix_sf, stress_sf_photo, stress_folder, stress_sf_graph,show,static_folder)
graphs (mesh, rev_degrees, stress_matrix_tf, stress_tf_photo, stress_folder, stress_tf_graph,show,static_folder)
graphs (mesh,rev_degrees,stress_matrix_vm_inside,stress_vm_inside_photo,stress_folder,stress_vm_inside_graph,show,static_folder)
graphs (mesh,rev_degrees,stress_matrix_vm_outside,stress_vm_outside_photo,stress_folder,stress_vm_outside_graph,show,static_folder)

#Displacements - Graphs
graphs (mesh, rev_degrees, displacement_matrix_v, displacement_v_photo, displacement_folder,displacement_v_graph,show,static_folder)
graphs (mesh, rev_degrees, displacement_matrix_w, displacement_w_photo, displacement_folder,displacement_w_graph,show,static_folder)
graphs (mesh, rev_degrees, displacement_matrix_theta, displacement_theta_photo, displacement_folder,displacement_theta_graph,show,static_folder)

#Strain - Graphs
graphs (mesh, rev_degrees, stress_matrix_epsilon_s, strain_epsilon_s_photo, strain_folder,strain_epsilon_s_graph,show,static_folder)
graphs (mesh, rev_degrees, stress_matrix_epsilon_theta, strain_epsilon_theta_photo, strain_folder,strain_epsilon_theta_graph,show,static_folder)
graphs (mesh, rev_degrees, stress_matrix_chi_s, strain_chi_s_photo, strain_folder,strain_chi_s_graph,show,static_folder)
graphs (mesh, rev_degrees, stress_matrix_chi_theta, strain_chi_theta_photo, strain_folder,strain_chi_theta_graph,show,static_folder)

#Safety Factor - Graphs
graphs (mesh, rev_degrees, safety_matrix_fsy, safety_fsy_photo, safety_folder,safety_fsy_graph,show,static_folder)
graphs (mesh, rev_degrees,safety_matrix_fsu,safety_fsu_photo,safety_folder,safety_fsu_graph,show,static_folder)

'''

# Stress - Graphs
graphs(mesh, points_matrix, stress_matrix_sd, stress_sd_photo, stress_folder, stress_sd_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_td, stress_td_photo, stress_folder, stress_td_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_sf, stress_sf_photo, stress_folder, stress_sf_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_tf, stress_tf_photo, stress_folder, stress_tf_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_vm_inside, stress_vm_inside_photo, stress_folder, stress_vm_inside_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_vm_outside, stress_vm_outside_photo, stress_folder, stress_vm_outside_graph, show, static_folder, loaded_image)

# Displacements - Graphs
graphs(mesh, points_matrix, displacement_matrix_v, displacement_v_photo, displacement_folder, displacement_v_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, displacement_matrix_w, displacement_w_photo, displacement_folder, displacement_w_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, displacement_matrix_theta, displacement_theta_photo, displacement_folder, displacement_theta_graph, show, static_folder, loaded_image)

# Strain - Graphs
graphs(mesh, points_matrix, stress_matrix_epsilon_s, strain_epsilon_s_photo, strain_folder, strain_epsilon_s_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_epsilon_theta, strain_epsilon_theta_photo, strain_folder, strain_epsilon_theta_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_chi_s, strain_chi_s_photo, strain_folder, strain_chi_s_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, stress_matrix_chi_theta, strain_chi_theta_photo, strain_folder, strain_chi_theta_graph, show, static_folder, loaded_image)

# Safety Factor - Graphs
graphs(mesh, points_matrix, safety_matrix_fsy, safety_fsy_photo, safety_folder, safety_fsy_graph, show, static_folder, loaded_image)
graphs(mesh, points_matrix, safety_matrix_fsu, safety_fsu_photo, safety_folder, safety_fsu_graph, show, static_folder, loaded_image)

'''

#Files 

#Stress - Files
files (stress_sd_file, stress_vect_sd, main_folder, stress_folder, static_folder,mesh)
files (stress_td_file, stress_vect_td, main_folder, stress_folder,static_folder,mesh)
files (stress_sf_file, stress_vect_sf, main_folder, stress_folder,static_folder,mesh)
files (stress_tf_file, stress_vect_tf, main_folder, stress_folder,static_folder,mesh)
files (stress_vm_inside_file,stress_vect_vm_inside, main_folder, stress_folder, static_folder, mesh)
files (stress_vm_outside_file,stress_vect_vm_outisde, main_folder, stress_folder, static_folder, mesh)

#Displacements - Files
files (displacement_v_file, displacement_vect_v, main_folder, displacement_folder,static_folder,mesh)
files (displacement_w_file, displacement_vect_w, main_folder, displacement_folder,static_folder,mesh)
files (displacement_theta_file, displacement_vect_theta, main_folder, displacement_folder,static_folder,mesh)

#Strain - Files
files (strain_epsilon_s_file, strain_vect_epsilon_s, main_folder, strain_folder,static_folder,mesh)
files (strain_epsilon_theta_file, strain_vect_epsilon_theta, main_folder, strain_folder,static_folder,mesh)
files (strain_chi_s_file, strain_vect_chi_s, main_folder, strain_folder,static_folder,mesh)
files (strain_chi_theta_file, strain_vect_chi_theta, main_folder, strain_folder,static_folder,mesh)

#Safety Factor - Graphs&Files
files (safety_fsy_file, safety_vect_fsy, main_folder, safety_folder,static_folder,mesh)
files(safety_fsu_file,safety_vect_fsu,main_folder,safety_folder,static_folder,mesh)

#Natural Frequencies 
nat_freqs (eig_vals, main_folder, natfreqs_folder, natfreqs_file,show,modal_folder)
 
#Time calculation
tf_output = time.time()

time_taken_analise = tf_analise - ti_analise
time_taken_output = tf_output - ti_output
time_total = time_taken_analise + time_taken_output

print()
print(f"Duração Análise: {round(time_taken_analise,2)} segundos")
print(f"Duração Output: {round(time_taken_output,2)} segundos")
print(f"Duração Total: {round(time_total,2)} segundos")

print()
print("-> Finish")
print()


