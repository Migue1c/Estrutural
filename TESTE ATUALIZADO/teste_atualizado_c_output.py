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
import matplotlib.gridspec as gridspec
import os
import warnings
import time

# Ignorar o aviso específico
warnings.filterwarnings("ignore", message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*")
warnings.filterwarnings("ignore", message="Conversion of an array with ndim > 0 to a scalar is deprecated.*")
warnings.filterwarnings("ignore", message="Casting complex values to real discards the imaginary part.*")

#Show Graphs?  
show = 0

#Slice angle to be poltted
rev_degrees = 360
rev_points = 250 

ti_analise = time.time()

def Mesh_Properties():
    
    file_name = 'Livro1.xlsx'
    # Number of Points to read
    df_num_rows  = pd.read_excel(file_name, sheet_name = 'Input', usecols = ['NumRowsRead'], nrows = 1)
    k            = int(df_num_rows.loc[0, 'NumRowsRead'])

    #Materials to read
    df_matcols  = pd.read_excel(file_name, sheet_name = 'Materials', usecols = [0], nrows = 1)
    m           = int(df_matcols.iloc[0, 0])
    matcols     = list(range(3, 3 + m))
    
    # Number of lines to read, for the loading
    df_loading_read = pd.read_excel(file_name, sheet_name = 'Loading', usecols = ['READ'], nrows = 1)
    k2              = int(df_loading_read.loc[0, 'READ'])
    
    # Reading the Input Data / Creating DataFrames
    df_read     = ['Points','z','r','thi','Conditions','Material','Conditions1','Nn', 'Loading', 'Discontinuity'] 
    df          = pd.read_excel(file_name, sheet_name = 'Input', usecols = df_read, nrows = k) 
    df_mat      = pd.read_excel(file_name, sheet_name = 'Materials', usecols = matcols, nrows = 7)
    
    # Matrix with the properties of the materials
    material = np.array(df_mat.values)

    # Creates a DataFrame with the same columns, but with no values
    empty_df        = pd.DataFrame(np.nan, index=[0], columns = df_read)
    
    #The first point can't be (0,0) due to mathematical issues
    if df.loc[0, 'r'] == 0:
        df.loc[0, 'r']  =  df.loc[0, 'r'] + 10**(-6) 

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
        
    # Creating the Mesh, with a higher density of elements close to nodes with Boundary Conditions or Discontinuity
        #Calculate the direction vector
        #Normalize the vector       
        #Determine the point and add to the Data Frame
    # User decides the type of mesh 
    df_mesh_type    = pd.read_excel(file_name, sheet_name = 'Input', usecols = ['Mesh Type'], nrows = 1)
    k5              = int(df_mesh_type.loc[0, 'Mesh Type'])
    
    if k5==1: # If it wants a mesh with non linear interpolation between the new nods
        i = 0
        while i < (len(df['Points'])):
            if not pd.isna(df.loc[i, 'Nn']) and (df.loc[i, 'Nn'] > 0):
                
                point1 = np.array(df.loc[i, ['r','z']].values)
                j = i + 1 + df.loc[i, 'Nn']
                point2 = np.array(df.loc[j, ['r','z']].values)
                
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
    else: # Interpolation Linear Type
        columns_interpolate     = ['z', 'r']
        df[columns_interpolate] = df[columns_interpolate].interpolate(method='linear')
    
    # Interpolation Linear Type of thicness and loading 
    columns_interpolate     = ['thi', 'Loading']
    df[columns_interpolate] = df[columns_interpolate].interpolate(method='linear')
    

    # Matrix with nodes coordinates 
    mesh = np.array(df[['z','r']].values)
    
    # Loading for each node
    pressure_nodes = np.array(df[['Loading']].values)
   
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

    # Boundary Condition Vector
    u_DOF = np.array(Boundary_Conditions["Value"].values)
    u_DOF = u_DOF.reshape((-1, 1))
  
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
                'thi'       : [np.nan],
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

        vpe.loc[i, 'thi'] = (df.loc[i, 'thi'] + df.loc[i+1, 'thi'])/2
    
    # Tranforming it to array
    vpe = np.array(vpe.values)

    """
    #Graphic of the points
    df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left')
    plt.show()
    """    

    ################### Static - Loading
    
    loading         = df['Loading'].to_numpy().reshape(-1)
    static_pressure = np.zeros(len(vpe)) # Builds a array filled with zeros with the length of the number of elements
    
    for i in range(0, len(vpe)):
        static_pressure[i] = (loading[i+1] + loading[i]) / 2 # The pressure aplied in the element is the average of the nodal values
    
    load_vct = np.zeros(3 * (len(vpe) + 1))
    for i in range(0, len(vpe)):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        hi = vpe[i, 2]
        p = static_pressure[i]
       
        v_carr = np.zeros(6)
        A11 = 0.5 * ri * (-np.sin(phi)) - (3 / 20) * np.sin(phi) ** 2 * hi
        A12 = 0.5 * ri * np.cos(phi) + (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 12) * ri + (1 / 30) * hi * np.sin(phi))
        A14 = 0.5 * ri * (-np.sin(phi)) - (7 / 20) * hi * np.sin(phi) ** 2
        A15 = 0.5 * ri * np.cos(phi) + (7 / 20) * hi * np.sin(phi) * np.cos(phi)
        A16 = hi * (-(1 / 12) * ri - (1 / 20) * hi * np.sin(phi))
        v_carr = 2*np.pi*hi*p*np.array([A11, A12, A13, A14, A15, A16])
        
        load_vct[3 * i:3 * i + 6] = load_vct[3 * i:3 * i + 6] + v_carr

    load_vct = np.reshape(load_vct,(-1,1))
    f_vect = load_vct
    
    
    
    
    
    ############################### Dynamic - Loading
    loading_cols = ['t_col', 'PressureCol', 'Add']
    df_loading  = pd.read_excel(file_name, sheet_name = 'Loading', usecols = loading_cols, nrows = k2 )
    empty_loading        = pd.DataFrame(np.nan, index=[0], columns = loading_cols)
    
    # Adding empty rows, with the number of time intervals necessary to have the delta_t specified by the user 
    i = 0
    while i < (len(df_loading['Add'])):
        if not pd.isna(df_loading.loc[i, 'Add']) and df_loading.loc[i, 'Add'] != 0:
            j=0
            while j < df_loading.loc[i, 'Add' ]:

                position    = i + 1
                result      = pd.concat([df_loading.iloc[:position], empty_loading, df_loading.iloc[position:]], ignore_index=True)
                df_loading          = result
                j += 1
        i +=1
    loading_cols     = ['t_col', 'PressureCol']
    df_loading[loading_cols] = df_loading[loading_cols].interpolate(method='cubic')
  
    t_col = np.array(df_loading[['t_col']].values)  #column vector
    p_col = np.array(df_loading[['PressureCol']].values) #column vector
   
    
    return mesh, u_DOF, vpe, material, pressure_nodes, t_col, p_col, f_vect

#Static
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
    kes = np.empty((6,6,ne), dtype=np.float64)
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        h = vpe[i, 2]
        D = elastM(i, vpe, mat)
        if simpson:
            I = np.empty((6, 6, ni+1), dtype=np.float64)
            for j, s1 in enumerate(np.linspace(0,1,ni+1)):
                r = ri + s1*h*np.sin(phi)
                B = Bmatrix(s1,i,r,phi, vpe)
                I[:,:,j] = B.T@D@B*(r)

            ke = 2*np.pi*h*sp.integrate.simpson(I, x=None, dx=1/ni, axis=-1)
        else:
            s1_ = lambda s: (s+1)/2
            r = lambda s: ri + s1_(s)*h*np.sin(phi)
            integrand = lambda s: Bmatrix(s1_(s),i,r(s),phi, vpe).T@D@Bmatrix(s1_(s),i,r(s),phi, vpe)*(r(s))
            ke = 2*np.pi*h*((5/9)*integrand(-np.sqrt(3/5))+(8/9)*integrand(0)+(5/9)*integrand(np.sqrt(3/5)))
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
        k_globalM = sp.sparse.bsr_array((data, (row, column)), shape=(3*(ne+1), 3*(ne+1)))#.toarray() 
        
    else:
        k_globalM = np.zeros((3*(ne+1), 3*(ne+1)), dtype=(np.float64))
        for i in range(0, ne):
            k_globalM[3*i:3*i+6,3*i:3*i+6] = k_globalM[3*i:3*i+6,3*i:3*i+6] + kes[:,:,i]

    return k_globalM
 
#Static Post-Process
def calculate_strains_stresses(displacements, vpe, mat):
    num_nodes = int(len(displacements)/3)
    num_elements = len(vpe)

    strains = np.zeros((num_nodes, 4))  # Matriz para armazenar as deformações de cada nó (epsilon_s, epsilon_theta, chi_s, chi_theta)
    for i in range(num_elements):
        R = vpe[i,0]
        phi = vpe[i,1]
        B = Bmatrix(0, i, R, phi, vpe)  # Obtém a matriz B para s1 = 0 
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

    #stress vai ser uma matriz em que cada coluna corresponde a [sigma_sd, sigma_td, sigma_sf, sigma_tf], em que d(dentro) e f(fora) 
    stress = np.zeros((num_nodes, 4))
    stress_memb = np.zeros((num_nodes, 2))
    for i in range(num_elements):
        t = vpe[i, 3]
        sigma_s = forças_N[i,0]/t
        sigma_t = forças_N[i,1]/t
        stress_memb[i, :] = [sigma_s, sigma_t]

        sigma_sd = sigma_s -  6*forças_N[i,2]/t**2
        sigma_sf = sigma_s +  6*forças_N[i,2]/t**2
        sigma_td = sigma_t -  6*forças_N[i,3]/t**2
        sigma_tf = sigma_t +  6*forças_N[i,3]/t**2
        stress[i, :] = [sigma_sd, sigma_td, sigma_sf, sigma_tf]
    i += 1
    sigma_s = forças_N[i,0]/t
    sigma_t = forças_N[i,1]/t
    stress_memb[i, :] = [sigma_s, sigma_t]

    sigma_sd = sigma_s - 6*forças_N[i,2]/t**2
    sigma_sf = sigma_s + 6*forças_N[i,2]/t**2
    sigma_td = sigma_t - 6*forças_N[i,3]/t**2
    sigma_tf = sigma_t + 6*forças_N[i,3]/t**2
    stress[i, :] = [sigma_sd, sigma_td, sigma_sf, sigma_tf]
    return strains, stress, stress_memb

def stress_VM(vpe, stress):      #matriz de duas colunas em que a primeira corresponde a dentro da casca e a segunda a fora da casca
    num_nodes = len(vpe) + 1
    num_elements = len(vpe)
    VM = np.zeros((num_nodes, 2))

    for i in range(num_elements):
        VM[i,0] = np.sqrt(stress[i,0]**2 -stress[i,0]*stress[i,2] + stress[i,2]**2)
        VM[i,1] = np.sqrt(stress[i,1]**2 -stress[i,1]*stress[i,3] + stress[i,3]**2)
    i += 1
    VM[i,0] = np.sqrt(stress[i,0]**2 -stress[i,0]*stress[i,2] + stress[i,2]**2)
    VM[i,1] = np.sqrt(stress[i,1]**2 -stress[i,1]*stress[i,3] + stress[i,3]**2)
    return VM

def FS(vpe, mat, VM, stress):     #FSy - deformação plastica  FSu - rutura
    VM = stress_VM(vpe, stress)
    von_mises = np.zeros(len(VM))
    for i, row in enumerate(VM):
        von_mises[i] += np.max(row)
    ne = len(vpe)
    FSy = np.empty((ne+1))
    FSU = np.empty((ne+1))
    for i in range(ne):
        if von_mises[i] == 0:
            FSy[i] = 10
        else:
            FSy[i] = mat[3, int(vpe[i, 4]) - 1] / von_mises[i]  
        FSc = 10
        FSt = FSc
        
        if np.any(stress[i,:] < 0):
            FSc = mat[5, int(vpe[i,4])-1] / np.min(stress[i,:])
        if np.any(stress[i,:] > 0):
            FSt = mat[4 ,int(vpe[i,4])-1] / np.max(stress[i,:])
        if np.abs(FSt) < np.abs(FSc):
            FSU[i] = FSt
        else:
            FSU[i] = FSc
    if von_mises[i+1] == 0:
        FSy[i+1] = 10
    else:
        FSy[i+1] = mat[3, int(vpe[i, 4]) - 1] / von_mises[i+1]  
    FSc = 10
    FSt = FSc
    if np.any(stress[i+1,:] < 0):
        FSc = mat[5, int(vpe[i,4])-1] / np.min(stress[i+1,:])
        #print(min(stress[i,:]))
    if np.any(stress[i+1,:] > 0):
        FSt = mat[4 ,int(vpe[i,4])-1] / np.max(stress[i+1,:])
        #print(FSt)
    if np.abs(FSt) < np.abs(FSc):
        FSU[i+1] = FSt
        #print(FSt)
    else:
        FSU[i+1] = FSc
        #print(FSc)
    #if np.any(stress[i+1,:] == 0):
        #fsu = 10
    return FSy, FSU

#Loading

#Dynamic Loading
def func_carr_t (funcoes, A, B, w, b, t_final, pi, util, t_col, p_col):
    if util[0] == 1:
        n = np.size(funcoes)
        dt = 0.01
        T = np.arange(0, seg_max, dt)
        seg_max = np.max(t_final)
        #print(t)
        #print(np.size(t))
        P = np.zeros(np.size(T))
        P[0]=pi[0]
        for i in range(0,n):
            if funcoes[i] == 1:
                A1 = A[0]      #ler do excel
                B1 = B[0]
                w1 = w[0]
                b1 = b[0]
                first_zero_index1 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t1 = t_final[0] - dt*first_zero_index1
                t_sin = np.arange(0, t1, dt)
                t_iter1 = np.size(t_sin)
                last_index1 = first_zero_index1 + t_iter1

                #print(t[first_zero_index1:last_index1])
                P_1 = np.zeros(t_iter1)
                for i in range (0,t_iter1):
                    t_1 = t_sin[i]
                    P_1[i] = A1*np.sin(w1*t_1+b1) + P[first_zero_index1-1] - A1*np.sin(w1*t_sin[0]+b1)
                #print(t_sin)
                P[first_zero_index1:last_index1] = P_1


            elif funcoes[i] == 2:
                A2 = A[1]
                B2 = B[1]
                first_zero_index2 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t2 = t_final[1] - first_zero_index2*dt
                t_exp = np.arange(0, t2, dt)
                t_iter2 = np.size(t_exp)
                last_index2 = first_zero_index2 + t_iter2
                #print(t[first_zero_index2:last_index2])
                P_2 = np.zeros(t_iter2)
                for i in range (0, t_iter2):
                    t_2 = t_exp[i]
                    P_2[i] = A2*np.exp(B2*t_2) + P[first_zero_index2-1] - 1
                #print(t_2)
                P[first_zero_index2:last_index2] = P_2


            elif funcoes[i] == 3:
                A3 = A[2]
                first_zero_index3 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t3 = t_final[2] - first_zero_index3 * dt
                t_lin = np.arange(0, t3, dt)
                t_iter3 = np.size(t_lin)
                last_index3 = first_zero_index3 + t_iter3
                #print(t[first_zero_index3:last_index3])
                P_3 = np.zeros(t_iter3)
                for i in range(0, t_iter3):
                    t_3 = t_lin[i]
                    P_3[i] = A3*t_3 + P[first_zero_index3-1]

                P[first_zero_index3:last_index3] = P_3

            elif funcoes[i] == 4:
                first_zero_index4 = np.where(P == 0)[0][0] if (P == 0).any() else None
                t4 = t_final[3] - first_zero_index4 * dt
                #print(t4)
                t_cst = np.arange(0, t4, dt)
                t_iter4 = np.size(t_cst)
                last_index4 = first_zero_index4 + t_iter4
                P_4 = np.zeros(t_iter4)
                for i in range(0, t_iter4):
                    P_4[i] = P[first_zero_index4-1]
                #print(P_4)
                P[first_zero_index4:last_index4] = P_4
    elif util[0] == 0:
        T = t_col
        P = p_col
    return P, T

def Carr_t(loading, t, T, P, press_max_est):

    p_col_adim = np.zeros(np.size(P))
    p_col_adim = P/press_max_est
    print(np.max(p_col_adim))
    P_t_adim = np.interp(t,T,p_col_adim)

    loading = loading * P_t_adim
    print(loading)
    return loading

#carregamento dinâmico só com teste estático
#(entra pressão -> saí vetor carregamento ; variação dos valores feita pela função da solução dinâmica)
def load_p(vpe, ne, P, pressure_nodes):

    max = np.amax(pressure_nodes)
    pressure_nodes1 = np.zeros(np.size(pressure_nodes))
    for i in range(0, ne+1):
        pressure_nodes1[i] = pressure_nodes[i] / max * P
    #pressão media
    press_medium = np.zeros(ne)
    for i in range(0, ne):
        press_medium[i] = (pressure_nodes1[i + 1] + pressure_nodes1[i]) / 2
    #vetor carregamento
    load_vct = np.zeros(3 * (ne + 1))
    for i in range(0, ne):
        phi = vpe[i, 1]
        ri = vpe[i, 0]
        hi = vpe[i, 2]
        p = press_medium[i]
        v_carr = np.zeros(6)
        A11 = 0.5 * ri * (-np.sin(phi)) - (3 / 20) * np.sin(phi) ** 2 * hi
        A12 = 0.5 * ri * np.cos(phi) + (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 12) * ri + (1 / 30) * hi * np.sin(phi))
        A14 = 0.5 * ri * (-np.sin(phi)) - (7 / 20) * hi * np.sin(phi) ** 2
        A15 = 0.5 * ri * np.cos(phi) + (7 / 20) * hi * np.sin(phi) * np.cos(phi)
        A16 = hi * (-(1 / 12) * ri - (1 / 20) * hi * np.sin(phi))
        v_carr = 2 * np.pi * hi * p * np.array([A11, A12, A13, A14, A15, A16])

        load_vct[3 * i:3 * i + 6] = load_vct[3 * i:3 * i + 6] + v_carr
    load_vct = load_vct.reshape((-1,1))
    return load_vct


#Modal Analsysis

def Mestacked(ne:int, vpe, mat, ni:int, simpson=True) -> np.ndarray:
    mes = np.empty((6,6,ne), dtype=float)
    for i in range(0, ne):
        rho = mat[0 , int(vpe[i, 4])-1,] # Specific mass for the material in the i-th element
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
        mes[:,:,i] = me
    return mes

def m_global(ne:int, vpe, mat, ni=1200, sparse=False) -> np.ndarray:
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

#def modal(eig_vals):
    natfreq = np.sort(np.sqrt(eig_vals))
    return natfreq

#Dynamic Analysis
def c_global(k_globalM, m_globalM, mode1:float, mode2:float, zeta1=0.08, zeta2=0.08):
    # Pressuposes that the global matrices k_globalM and m_globalM are already reduced

    fraction = 2*mode1*mode2/(mode2**2-mode1**2)
    alfa = fraction*(mode2*zeta1-mode1*zeta2)
    beta = fraction*(zeta2/mode1-zeta1/mode2)

    c_globalM = alfa*m_globalM + beta*k_globalM
    return c_globalM

#Dynamiic Post Process
def dyn_post_proc(displacements, vpe, material):
    # Getting number of nodes and instants of time
    n_nodes = len(vpe) + 1
    time_instas = displacements.shape[1]

    # Initializing empty variables
    strains_3 = np.empty((n_nodes, 4, time_instas))
    stress_3 = np.empty((n_nodes, 4, time_instas))
    stress_memb_3 = np.empty((n_nodes, 2, time_instas))
    t_VM = np.empty((n_nodes, 2, time_instas))
    fsy = np.empty((n_nodes, 1, time_instas))
    fsu = np.empty((n_nodes, 1, time_instas))

    # Filling them with values for each instant of the dynamic simulation
    for j in range(0,time_instas):      # for each displacement column, aka each time instant
        dis = displacements[:,j].reshape((-1,1))
        strain, stress, stress_memb = calculate_strains_stresses(dis, vpe, material)
        strains_3[:,:,j] = strain        # strains
        stress_3[:,:,j] = stress       # direct stresses inside/outside of shell
        stress_memb_3[:,:,j] = stress_memb     # direct membrane stresses, direct stresses in midle plane
        t_VM[:,:,j] = stress_VM(vpe, stress)      # von Mises stresses inside/outside
        fsy[:,0,j], fsu[:,0,j] = FS(vpe, material, t_VM[:,:,j], stress)     # yield safety factor, ultimate safety factor



    return strains_3, stress_3, stress_memb_3, t_VM, fsy, fsu

#Solution

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

    #re-add zeros to the eigenvectors matrix
    eig_vect = RdfMatrix(eig_vect, u_DOF)

    #sort values 
    guide_vect = np.argsort(eig_vals)
    natfreq = np.sort(np.sqrt(eig_vals))

    freq1= natfreq[0]       #used to calculate damping matrix
    freq2= natfreq[1]
    natfreq = natfreq/(2*np.pi) #convert to hertz

    #sort vector
    new_mtx = np.zeros((len(eig_vect),len(guide_vect)))
    n=0
    for i in guide_vect:
        new_mtx[:,n] = eig_vect[:,i]
        n += 1
    eig_vect = new_mtx

    #print('Valores Próprios:', natfreq)
    return natfreq, eig_vect, freq1, freq2

#Dinamic Solution:
#STATIC TEST VERSION ONLY
def DynamicSolver(k:np.ndarray, m:np.ndarray, c:np.ndarray, u_DOF:np.ndarray, t_col, p_col, vpe, ne, pressure_nodes):

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
        #Linear Acceleration Method:wr 
        gamma = 1/2
        beta = 1/6
    
    #time constraints
    l = t_col.shape[0] - 1
    tk = t_col[0,0]
    t_final = t_col[l,0]
    fg = 0

    f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
    f = RedMatrix(f, u_DOF)
    x_0_d2 = sp.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))


    while tk < t_final :
        
        #Starting value [x_d2_(0)]
        #x_0_d2 = sp.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))

        #Time increment:
        tk0 = tk
        fg += 1
        tk = t_col[fg,0]
        delta_t = tk - tk0

        #Force vector for current tk
        f = load_p(vpe, ne, p_col[fg,0], pressure_nodes)
        f = RedMatrix(f, u_DOF)
        
        #Prediction:
        x_1_d = x_0_d + (1 - gamma) * delta_t * x_0_d2
        x_1 = x_0 + delta_t * x_0_d + (0.5 - beta)*(delta_t**2) * x_0_d2
        
        #Equilibrium eqs.:
        s = m + (gamma * delta_t * c) + (beta * (delta_t**2) * k)
        x_1_d2 = sp.linalg.inv(s) @ (f - (c @ x_0_d) - (k @ x_0) )

        #Correction:
        x_1_d = x_1_d + (delta_t * gamma * x_1_d2)
        x_1 = x_1 + ((delta_t**2) * beta * x_1_d2)

        #store values in matrices
        matrix_u = np.append(matrix_u, x_1, axis=1)
        matrix_ud = np.append(matrix_ud, x_1_d, axis=1)
        matrix_ud2 = np.append(matrix_ud2, x_1_d2, axis=1)

        #reset starting values for next iteration:
        x_0 = x_1
        x_0_d = x_1_d
        x_0_d2 = x_1_d2

    #add lines with zeros to the matrices
    matrix_u = RdfMatrix(matrix_u, u_DOF)
    matrix_ud = RdfMatrix(matrix_ud, u_DOF)
    matrix_ud2 = RdfMatrix(matrix_ud2, u_DOF)

    return matrix_u, matrix_ud, matrix_ud2

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
        x_0_d2 = x_1_d2

    #add lines with zeros to the matrices
    matrix_u = RdfMatrix(matrix_u, u_DOF)
    matrix_ud = RdfMatrix(matrix_ud, u_DOF)
    matrix_ud2 = RdfMatrix(matrix_ud2, u_DOF)

    return matrix_u, matrix_ud, matrix_ud2


#Output

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
    
def files (file_name, metric_vect, main_folder, metric_folder,analysis_folder,points): #Save Array data
    #Full path to save the file inside the folder
    file_path = os.path.join(main_folder,analysis_folder,metric_folder, file_name)
    #Write file inside the folder
    with open(file_path, 'w') as file:
        for point, metric_value in zip(points[:,0], metric_vect):
            formatted_point = "{:.3f}".format(point)  # Formata o ponto para ter no máximo 3 casas decimais
            file.write(f'{formatted_point} {metric_value}\n')

def tecplot_exporter(file_path,rev_angle, divisions, mesh, u_global, strains, stress, stress_memb, t_VM, fsy, fsu, deformation):
    
    if deformation != 0:
        n_nodes = len(mesh)
        def_mesh = np.copy(mesh)
        def_mesh[:,0] += u_global[0::3,0]*(deformation*100)
        def_mesh[:,1] += u_global[1::3,0]*(deformation*100)
    else:
        def_mesh = np.copy(mesh)
        def_mesh[:,0] += u_global[0::3,0]
        def_mesh[:,1] += u_global[1::3,0]

    rev_angle = math.radians(rev_angle)
    n_nodes = len(mesh)
    angles = np.linspace(0, rev_angle , divisions, endpoint=True)

    x = np.empty([n_nodes, divisions])
    y = np.empty([n_nodes, divisions])
    for i in range(0, n_nodes):
        x[i, :] = def_mesh[i,1]*np.cos(angles)
        y[i, :] = def_mesh[i,1]*np.sin(angles)

    #Write file inside the folder
    with open(file_path, 'w') as file:
        file.write("TITLE = \"Shell\"\n")
        #file.write("VARIABLES = x, y, z, v, w, theta, es, et, xs, xt, ssd, std, ssf, stf, VM_d, VM_f, fsy, fsu\n")
        file.write("VARIABLES = x, y, z, v, w, theta, es, et, xs, xt, ssd, std, ssf, stf, ss_memb, st_memb, VM_d, VM_f,  fsy, fsu\n")
        if deformation == 0:
            file.write(f"ZONE T=\"undeformed\", I={n_nodes:04d} J={divisions:04d}\n")
        else:
            file.write(f"ZONE T=\"deformed\", I={n_nodes:04d} J={divisions:04d}\n")
        for i in range(0, divisions):
            for j in range(0, n_nodes):
                file.write(f'{float(x[j,i]):.7e}  {float(y[j,i]):.7e}  {float(mesh[j,0]):.7e}  {float(u_global[3*j,0]):.7e}  {float(u_global[3*j+1,0]):.7e}  {float(u_global[3*j+2,0]):.7e}  {float(strains[j,0]):.7e}  {float(strains[j,1]):.7e}  {float(strains[j,2]):.7e}  {float(strains[j,3]):.7e}  {float(stress[j,0]):.7e}  {float(stress[j,1]):.7e}  {float(stress[j,2]):.7e}  {float(stress[j,3]):.7e}  {float(stress_memb[j,0]):.7e}  {float(stress_memb[j,1]):.7e}  {float(t_VM[j,0]):.7e}  {float(t_VM[j,1]):.7e}  {float(fsy[j]):.7e}  {float(fsu[j]):.7e}\n')

def tecplot_exporter_dynamic(rev_angle, divisions, mesh, u_global, strains, stress, stress_memb, t_VM, fsy, fsu, t_col,  deformation):
   if deformation > 0:
        for i in range(u_global.shape[1]):
            u = u_global[:,i].reshape((-1,1))
            tecplot_exporter((f'FEM Analysis - Data\Dynamic Analysis\output_tecplot_3d(t={t_col[i,0]:.3f}s).txt'),rev_angle, divisions, mesh, u, strains[:,:,i], stress[:,:,i], stress_memb[:,:,i], t_VM[:,:,i], fsy[:,:,i], fsu[:,:,i], deformation)
    
        else:
            for i in range(u_global.shape[1]):
                u = u_global[:,i].reshape((-1,1))
                tecplot_exporter((f'FEM Analysis - Data\Dynamic Analysis\output_tecplot_3d_deformed(t={t_col[i,0]:.3f}s).txt'),rev_angle, divisions, mesh, u, strains[:,:,i], stress[:,:,i], stress_memb[:,:,i], t_VM[:,:,i], fsy[:,:,i], fsu[:,:,i], deformation)

def tecplot_exporter_dynamic_zones(file_path,rev_angle, divisions, mesh,t_col, u_global, strains_zone, stress_zone, stress_memb_zone, t_VM_zone, fsy_zone, fsu_zone, deformation):
    
    rev_angle = math.radians(rev_angle)
    n_nodes = len(mesh)
    angles = np.linspace(0, rev_angle , divisions, endpoint=True)

    x = np.empty([n_nodes, divisions])
    y = np.empty([n_nodes, divisions])
    
    #Write file inside the folder
    with open(file_path, 'w') as file:
        file.write("TITLE = \"Shell\"\n")
        file.write("VARIABLES = x, y, z, v, w, theta, es, et, xs, xt, ssd, std, ssf, stf, ss_memb, st_memb, VM_d, VM_f,  fsy, fsu\n")

        for k in range(u_global.shape[1]):
            if deformation != 0:
                n_nodes = len(mesh)
                def_mesh = np.copy(mesh)
                def_mesh[:,0] += u_global[0::3,k]*(deformation*100)
                def_mesh[:,1] += u_global[1::3,k]*(deformation*100)
            else:
                def_mesh = np.copy(mesh)
                def_mesh[:,0] += u_global[0::3,k]
                def_mesh[:,1] += u_global[1::3,k]
            file.write(f"ZONE T=\"{t_col[k,0]:.3f}\", I={n_nodes:04d} J={divisions:04d}\n")
            u = u_global[:,k].reshape((-1,1))
            strains = strains_zone[:,:,k]
            stress = stress_zone[:,:,k]
            stress_memb = stress_memb_zone[:,:,k]
            t_VM = t_VM_zone[:,:,k]
            fsy = fsy_zone[:,:,k]
            fsu = fsu_zone[:,:,k] 
            for l in range(0, n_nodes):
                x[l, :] = def_mesh[l,1]*np.cos(angles)
                y[l, :] = def_mesh[l,1]*np.sin(angles)   

            for i in range(0, divisions):
                for j in range(0, n_nodes):
                    file.write(f'{float(x[j,i]):.7e}  {float(y[j,i]):.7e}  {float(def_mesh[j,0]):.7e}  {float(u[3*j,0]):.7e}  {float(u[3*j+1,0]):.7e}  {float(u[3*j+2,0]):.7e}  {float(strains[j,0]):.7e}  {float(strains[j,1]):.7e}  {float(strains[j,2]):.7e}  {float(strains[j,3]):.7e}  {float(stress[j,0]):.7e}  {float(stress[j,1]):.7e}  {float(stress[j,2]):.7e}  {float(stress[j,3]):.7e}  {float(stress_memb[j,0]):.7e}  {float(stress_memb[j,1]):.7e}  {float(t_VM[j,0]):.7e}  {float(t_VM[j,1]):.7e}  {float(fsy[j]):.7e}  {float(fsu[j]):.7e}\n')

def teplot_exporter_2d(file_path, t_col, u_global_native, strains, stress, stress_memb, t_VM, fsy, fsu):
    
    vars_name = ['v', 'w', 'theta', 'es', 'et', 'xs', 'xt', 'ssd', 'std', 'ssf', 'stf', 'ss_memb', 'st_memb', 'VM_d', 'VM_f', 'fsy', 'fsu']
    vars = len(vars_name)
    nn = len(strains)
    time_steps = len(t_col)

    u_global = np.empty((nn,3,time_steps))
    u_global [:,0,:] = u_global_native[0::3, :]
    u_global [:,1,:] = u_global_native[1::3, :]
    u_global [:,2,:] = u_global_native[2::3, :]

    u_global = np.hstack((u_global,strains, stress, stress_memb, t_VM, fsy, fsu)) #caso dê erro fsy/fsu é preciso passar para matriz coluna e nao deixar apenas como vetor

    data_max = np.empty((time_steps, vars))
    for i in range(time_steps):
        for j in range(vars):
            data_max[i,j] = np.max(u_global[:,j,i])

    #Write file inside the folder
    with open(file_path, 'w') as file:
        file.write("TITLE = \"2D Graphs\"\n")
        #file.write(f"VARIABLES = t, v, w, theta, es, et, xs, xt, ssd, std, ssf, stf, ss_memb, st_memb, VM_d, VM_f, fsy, fsu\n")
        file.write(f"VARIABLES = t, v, w, theta, es, et\n")
        for i in range(0, vars):
            #file.write(f"VARIABLES = t, {vars_name[i]}\n")
            file.write(f"ZONE T=\"{vars_name[i]}\", I={nn:04d} F=POINT\n")
            for j in range(0, time_steps):
                file.write(f'{t_col[j,0]:.7e}  {data_max[j,i]:.7e}\n') 

def nat_freqs(natural_frequencies, main_folder, metric_folder, file_name, show, analysis_folder):
    
    modes_graph = 10

    modes_number = len(natural_frequencies)
    modes = [i + 1 for i in range(modes_number)]

    # File path for saving all modes
    all_modes_file_path = os.path.join(main_folder, analysis_folder, metric_folder, file_name)
    
    # Writing data to a text file (all modes)
    with open(all_modes_file_path, 'w') as file:
        file.write("Natural_frequencies\n")
        for z, freq in zip(modes, natural_frequencies):
            file.write(f"{freq}\n")

    # Selecting only the specified number of modes for the plot
    modes_to_show = modes[:modes_graph]
    frequencies_to_show = natural_frequencies[:modes_graph]

    # File path for saving the plot with selected modes
    plot_path = os.path.join(main_folder, analysis_folder, metric_folder, "natural_frequencies_graph.png")

    # Create a subplot grid with reduced horizontal spacing
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 0.5], wspace=0.1)  # 2 columns, 1.5:0.5 width ratio, reduced spacing

    # Plot the bar chart with increased size for the specified modes
    ax1 = plt.subplot(gs[0])
    ax1.bar(modes_to_show, frequencies_to_show, color='r', align='center', alpha=0.7, width=0.2)
    ax1.set_xlabel('Vibration Mode')
    ax1.set_ylabel('Natural Frequency [Hz]')
    ax1.set_title('Natural Frequencies Graph')
    #ax1.grid(False)
    ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    ax1.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)
    

    # Set ticks to show all modes on the x-axis
    plt.xticks(modes_to_show)

    # Plot the table with a zoom effect
    ax2 = plt.subplot(gs[1])
    ax2.axis('off')  # Turn off the axis for the table

    # Adjust the position of the table to the left
    table_data = [[f'{mode}', f'{freq:.2f} Hz'] for mode, freq in zip(modes_to_show, frequencies_to_show)]
    table = ax2.table(cellText=table_data, loc='center left', cellLoc='center', colLabels=['Mode', 'Frequency'], cellColours=[['#f0f0f0']*2]*len(table_data))
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    # Zoom effect: Increase the scale of the table
    table.scale(2, 2)  # Adjust the scale factor as needed
    table.auto_set_column_width([0, 1])  # Adjust column width
    
    # Save the plot
    plt.savefig(plot_path, bbox_inches='tight')

    # Show the plot if requested
    if show == 1:
        plt.show()

    plt.close()


#############################################################################################################################################
print()
print('File openning -> On')
ti_mesh = time.time()

#File Reading
mesh, u_DOF, vpe, material, pressure_nodes, t_col, p_col, f_vect = Mesh_Properties()

tf_mesh = time.time()
print('File openning -> Completed') 
print()

#############################################################################################################################################

ti_static = time.time()
print('Static Anlysis -> On')
#Static Analysis
#K matrix
k = k_global(len(vpe), vpe, material)    #calculo matriz K

#Static Solution & Static Post Process
u_global = StaticSolver(k, f_vect, u_DOF)                                                       #Compute of displacements              
strains, stress, stress_memb = calculate_strains_stresses(u_global, vpe, material)              #Compute of strains and direct stresses (e_s, e_th, x_s, x_th)
t_VM = stress_VM(vpe, stress)                                                                   #Compute of Von Mises Stress (t_s_d, t_th_d, t_s_f, t_th_f)
fsy, fsu = FS(vpe, material, t_VM, stress)                                                      #Compute of Safety Factor (fsy-cedencia, fsu-rutura)

tf_static = time.time()
print('Static Analysis -> Completed')
print()

#############################################################################################################################################

ti_modal = time.time()
print('Modal Anlysis -> On')

#Modal Analisys
#M Matrix
m_gl = m_global(len(vpe), vpe, material, ni=1200, sparse=False) #calculo matriz M

#Modal solution & Modal Post Process
natfreq, eig_vect, freq1, freq2 = ModalSolver(k, m_gl, u_DOF)                 #calculo valores e vetores próprios

tf_modal = time.time()
print('Modal Analysis -> Completed')
print()

#############################################################################################################################################

ti_dynamic = time.time()
print('Dynamic Anlysis -> On')

#Dynamic Analysis
#C Matrix
c = c_global(k, m_gl, freq1, freq2)                             #calculo matriz C

#Dynamic solution & Modal Post Process
u_global_matrix, matrix_v, matrix_a = DynamicSolverV2(k, m_gl, c, u_DOF, t_col, p_col, vpe, len(vpe), pressure_nodes)
strains_d, stress_d, stress_memb_d, t_VM_d, fsy_d, fsu_d = dyn_post_proc(u_global_matrix, vpe, material)

tf_dynamic = time.time()
print('Dynamic Analysis -> Completed')
print()

#############################################################################################################################################


#Output
print()
print("Output -> On")
print()

ti_output = time.time()

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

#STRESS DATA
#Stress_SD - Output
stress_sd_file = "Stress-SD.txt"    #File Name
stress_vect_sd= stress[:,0]
#Stress_SD - Output
stress_td_file = "Stress-TD.txt"    #File Name
stress_vect_td= stress[:,1]
#Stress_SF - Output
stress_sf_file = "Stress-SF.txt"    #File Name
stress_vect_sf= stress[:,2]
#Stress_TF - Output
stress_tf_file = "Stress-TF.txt"    #File Name
stress_vect_tf= stress[:,3]
#Stress Von Mises Inside - Output
stress_vm_inside_file =  "Stress VM - Inside.txt"   #File Name
stress_vect_vm_inside = t_VM[:,0]
#Stress Von Mises Inside - Output
stress_vm_outside_file =  "Stress VM - Outside.txt" #File Name
stress_vect_vm_outisde = t_VM[:,1]


#Displacement
#Displacement V - Output
displacement_v_file = "Displacement-V.txt"  #File Name
displacement_vect_v = u_global[0::3,0]
#Displacement W - Output
displacement_w_file = "Displacement-W.txt"  #File Name
displacement_vect_w = u_global[1::3,0]
#Displacement W - Output
displacement_theta_file = "Displacement-Theta.txt"  #File Name
displacement_vect_theta = u_global[2::3,0]

#STRAIN DATA
#Strain: Epsilon_S - Output
strain_epsilon_s_file = "Strain-Epsilon S.txt"  #File Name
strain_vect_epsilon_s= strains[:,0]
#Strain: Epsilon_Theta - Output
strain_epsilon_theta_file = "Strain-Epsilon Theta.txt"  #File Name
strain_vect_epsilon_theta= strains[:,1]
#Strain: Chi_S - Output
strain_chi_s_file = "Strain-Chi S.txt"  #File Name
strain_vect_chi_s= strains[:,2]
#Strain: Chi_Theta - Output
strain_chi_theta_file = "Strain-Chi Theta.txt"  #File Name
strain_vect_chi_theta= strains[:,3]

#FSy - deformação plastica  FSu - rutura
#Safety Factor: Fsy - Output
safety_fsy_file = "Safety Factor FSy.txt"   #File Name
safety_vect_fsy= fsy
#Safety Factor: Fsy - Output
safety_fsu_file = "Safety Factor FSu.txt"   #File Name
safety_vect_fsu= fsu

#Natural Frequencies
natfreqs_file = "Natural Frequencies.txt"   #File Name

#Plots/Figures

#Creation of folders to add files from the analysis (Mandatory)
folders_creator(main_folder,stress_folder,strain_folder,natfreqs_folder,static_folder,modal_folder,dynamic_folder,displacement_folder,safety_folder)

#Geometry Plot
#geometry_plot(mesh,rev_degrees,main_folder,geometry_photo, show)

#Tecplot data exporter for 3D (Static)
tecplot_exporter('FEM Analysis - Data\Static Analysis\output_export_tecplot_3d.txt',rev_degrees, rev_points, mesh, u_global, strains, stress, stress_memb, t_VM, fsy, fsu,0)
tecplot_exporter('FEM Analysis - Data\Static Analysis\output_export_tecplot_3d_deformed.txt',rev_degrees, rev_points, mesh, u_global, strains, stress, stress_memb, t_VM, fsy, fsu,4)

#Tecplot data exporter for 3D (Dynamic)
tecplot_exporter_dynamic_zones('FEM Analysis - Data\Dynamic Analysis\output_export_dynamic_deformed_tecplot_3d.txt',rev_degrees, rev_points, mesh,t_col, u_global_matrix, strains_d, stress_d, stress_memb_d, t_VM_d, fsy_d, fsu_d, 4)

#Tecplot data export for 2D (Dynamic)
teplot_exporter_2d('FEM Analysis - Data\Dynamic Analysis\output_export_dynamic_tecplot_2d.txt', t_col, u_global_matrix, strains_d, stress_d, stress_memb_d, t_VM_d, fsy_d, fsu_d)

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
nat_freqs (natfreq, main_folder, natfreqs_folder, natfreqs_file,show,modal_folder)
 
#Time calculation
tf_output = time.time()

time_taken_mesh = tf_mesh - ti_mesh
time_taken_static = tf_static - ti_static
time_taken_modal = tf_modal - ti_modal
time_taken_dynamic = tf_dynamic - ti_dynamic
time_taken_output = tf_output - ti_output
time_total = tf_output - ti_mesh



print()
print(f"File openning: {round(time_taken_mesh,2)} seconds")
print(f"Static Analysis: {round(time_taken_static,2)} seconds")
print(f"Modal Analysis: {round(time_taken_modal,2)} seconds")
print(f"Dynamic Analysis: {round(time_taken_dynamic,2)} seconds")
print(f"Output: {round(time_taken_output,2)} seconds")
print(f"Run time: {round(time_total,2)} seconds")

print()
print("-> Finish")
print()









#############################################################################################################################################



