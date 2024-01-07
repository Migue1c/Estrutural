import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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
    
    P_col = np.array(df_loading[['p1']].values) #column vector
    #print(P_col)
    
    # Matrix with the properties of the materials
    material = np.array(df_mat.values)
    
    #print(material)


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
    
    
    # Creating the Mesh, with a higher density of elements close to nodes with Boundary Conditions or Discontinuity
    
        #Calcular o vetor de direção entre os nós iniciais
        #Normalizar o vetor        
        #Calcular o ponto e adicionar ao DataFrame
            
    i = 0
    while i < (len(df['Points'])):
        
        if (df.loc[i, 'Nn'] > 0):
            
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
                while k < df.loc[i, 'Nn']:
                    
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

    print(df)


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

    """
    #Graphic of the points
    df.plot(x='r', y='z', marker='o', linestyle='-', color='k', label='Nós')
    plt.gca().invert_yaxis()
    plt.legend(loc='center left')
    plt.show()
    """
    
    return mesh, u_DOF, vpe, material, pressure_nodes, t_col, P_col


Mesh_Properties()


def AnalysisType():
    
    # Reading the analysis the user wants
    analysis_columns = pd.read_excel('Livro1.xlsx', sheet_name = 'Input', usecols = ['Static', 'Modal', 'Dynamic'], nrows = 1)
    
    print(analysis_columns)
    
AnalysisType()
    
    
    
    








