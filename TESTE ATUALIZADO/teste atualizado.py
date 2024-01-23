import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sp

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
        
    if 1==1: ################## A alterar, para variar com os dados inseridos no excel
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
    
    if 1==0: ################## A alterar, para variar com os dados inseridos no excel
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

    
    #Graphic of the points
    df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left')
    plt.show()
    
    
    #print(material)
    #print(vpe)
    
    return mesh, u_DOF, vpe, material, pressure_nodes, t_col, p_col



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
 
#POS-PROCESSAMENTO
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
    tensoes_memb = np.zeros((num_nodes, 2))
    for i in range(num_elements):
        t = vpe[i, 3]
        sigma_s = forças_N[i,0]/t
        sigma_t = forças_N[i,1]/t
        tensoes_memb[i, :] = [sigma_s, sigma_t]

        sigma_sd = sigma_s -  6*forças_N[i,2]/t**2
        sigma_sf = sigma_s +  6*forças_N[i,2]/t**2
        sigma_td = sigma_t -  6*forças_N[i,3]/t**2
        sigma_tf = sigma_t +  6*forças_N[i,3]/t**2
        tensoes_N[i, :] = [sigma_sd, sigma_td, sigma_sf, sigma_tf]
    i += 1
    sigma_s = forças_N[i,0]/t
    sigma_t = forças_N[i,1]/t
    tensoes_memb[i, :] = [sigma_s, sigma_t]

    sigma_sd = sigma_s - 6*forças_N[i,2]/t**2
    sigma_sf = sigma_s + 6*forças_N[i,2]/t**2
    sigma_td = sigma_t - 6*forças_N[i,3]/t**2
    sigma_tf = sigma_t + 6*forças_N[i,3]/t**2
    tensoes_N[i, :] = [sigma_sd, sigma_td, sigma_sf, sigma_tf]
    return strains, tensoes_N, tensoes_memb

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



#QUITÉRIO
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
        p = pressure[i]
        #print(phi, ri, hi, p)
        v_carr = np.zeros(6)
        A11 = 0.5 * ri * (-np.sin(phi)) - (3 / 20) * np.sin(phi) ** 2 * hi
        A12 = 0.5 * ri * np.cos(phi) + (3 / 20) * np.sin(phi) * np.cos(phi) * hi
        A13 = hi * ((1 / 12) * ri + (1 / 30) * hi * np.sin(phi))
        A14 = 0.5 * ri * (-np.sin(phi)) - (7 / 20) * hi * np.sin(phi) ** 2
        A15 = 0.5 * ri * np.cos(phi) + (7 / 20) * hi * np.sin(phi) * np.cos(phi)
        A16 = hi * (-(1 / 12) * ri - (1 / 20) * hi * np.sin(phi))
        v_carr = 2*np.pi*hi*p*np.array([A11, A12, A13, A14, A15, A16])
        
        load_vct[3 * i:3 * i + 6] = load_vct[3 * i:3 * i + 6] + v_carr

    return load_vct

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
                    P_2[i] = A2*np.exp(B2*t_2) * P[first_zero_index2-1]
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


def load_p(vpe, ne, P, pressure_nodes):
    max = np.amax(pressure_nodes)
    for i in range (0,ne):
        pressure_nodes = pressure_nodes / max * P

    
    

#BOMBAS
#MODAL

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

#def modal(eig_vals):
    natfreq = np.sort(np.sqrt(eig_vals))
    return natfreq


#ESTEVES
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
#loading, t_col, P_col
#inputs need change
def DinamicSolver(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float):

    #Reduce Matrices
    k = RedMatrix(k, u_DOF)
    m = RedMatrix(m, u_DOF)
    c = RedMatrix(c, u_DOF)

    #Define starting values vector (reduced)
    l = k.shape[0]
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
    
    while tk <= t_final :
    
        #Force vector for current tk
        #f = Carr_t(loading, tk, t_col, P_col)
        #f = RedMatrix(f, u_DOF)

        #Starting value [x_d2_(0)]
        x_0_d2 = np.linalg.inv(m) @ (f - (c @ x_0_d ) - (k @ x_0))
        
        #Time increment:
        tk += delta_t

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



'''
seg_max = np.max(t_final)

    if util[0] == 1:
        P = func_carr_t(funcoes, A, B, w, b, t_final, pi, seg_max)[0]
        T = func_carr_t(funcoes, A, B, w, b, t_final, pi, seg_max)[1]
    elif util[0] == 0:
        P = ler pelo olim
        T = ""

#carregamento Dinamico
press_max_est = np.max(pressure_nodes)
'''


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
strains, tensoes_N, tensoes_memb = calculate_strains_stresses(u_global, vpe, material)    #calculo das extensões e tensões diretas (e_s, e_th, x_s, x_th)
t_VM = tensões_VM(u_global, vpe, tensoes_N)                 #calculo das tensões de von-misses (t_s_d, t_th_d, t_s_f, t_th_f)
fsy, fsu = FS(u_global, vpe, material, t_VM, tensoes_N)     #calculo dos fatores de segurança (fsy-cedencia, fsu-rutura)
#print("strains:\n",strains)
#print("tensões:\n",tensoes_N)
#print("tensões membrana:\n",tensoes_memb)
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
natfreq, eig_vect = ModalSolver(k, m_gl, u_DOF)               #calculo valores e vetores próprios
#print("valores proprios:\n",natfreq)                      
#print("vetores proprios:\n",eig_vect)                   
#print("freq. natural 1:\n",natfreq1)
#print("freq. natural 2:\n",natfreq2)


#ANÁLISE DINÂMICA
#MATRIZ C
c = c_global(k, m_gl, natfreq[0], natfreq[1])                   #calculo matriz C
#c_df = pd.DataFrame(c)                                      #converter pra dataframe
#c_df.to_excel('c.xlsx', index=False)                        #guardar DF no excel
#print(c)

#DinamicSolver(m:np.ndarray, c:np.ndarray, k:np.ndarray, f:np.ndarray, x_0:np.ndarray, x_0_d:np.ndarray, u_DOF:np.ndarray, tk:float, delta_t:float, t_final:float, loading, t_col, P_col)