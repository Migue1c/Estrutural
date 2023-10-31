import numpy as np
import sympy as sp

##Elasticity matrix for the isotropic axisymmetrically loaded 

E = 200
upsilon = 0.2
t = 0.1

def elastM(E,t,upsilon):
    D = (E*t)/(1-upsilon**2)*np.array([[1,upsilon, 0, 0],[upsilon, 1, 0, 0],[0, 0, (t**2/12), upsilon*(t**2/12)],[0, 0, upsilon*(t**2/12), (t**2/12)]])
    return D

D = elastM(E,t,upsilon)
print("Elasticity Matrix:\n",D)

##Transformation matrix

phi = 30*np.pi/180

def transM(phi):
    T = np.array([[np.cos(phi), np.sin(phi), 0],[-np.sin(phi), np.cos(phi), 0],[0, 0, 1]])
    return T

T = transM(phi)
print("Transformation Matrix:\n",T)

##Mathematical model calculations for element stiffness matrix -> refer to book explanation for more information

    #v = a1 + a2*s                          system of equations for element aproximation
    #w = a3 + a4*s + a5*a**2 + a6*a**3
    #s1 = s/h                               s1=0 if index=i for node 1 and s2=1 if index=j for node 2

    #L = np.array([[1,s1,0,0,0,0],[0,0,0,1,s1,s1**2,s1**3]])    #conversion to matrix system
    #a = np.array([a1,a2,a3,a4,a5,a6])
    #dw_ds = np.array([0,0,0,1,2*s1,3*s1**2])                   #third row with node rotation

s1 = sp.Symbol("s1")     
s =  sp.Symbol("s")
h = sp.Symbol("h")
r = sp.Symbol("r")
cosphi = sp.Symbol("cosphi")
sinphi = sp.Symbol("sinphi")
L = sp.Matrix([[1,s1,0,0,0,0],[0,0,1,s1,s1**2,s1**3]])      #Matrix L from above
dw_dsk = []                             #opens vector for thrid row of Matrix Ci,j
for i in range(6):
    dw_dsk.append(sp.diff(L[1,i]))      #populates vector with dw_ds
Ck  = np.vstack((L,dw_dsk))             #stacks vector to form 3x6 Matrix Ck
Ck = sp.Matrix(Ck)                      #converts numpy matrix to sympy matrix
C = sp.zeros(6)                         #defines 6x6 Matrix 
C[:3,:6] = Ck.subs(s1,0)            #up to row 3 (0,1,2) up to column 6 (0,1,2,3,4,5), populate with Ci, s1=0
C[3:,:6] = Ck.subs(s1,1)            #from row 3 (3,4,5) up to column 6 (0,1,2,3,4,5), populate with Cj, s1=1
C_inv = C.inv()                         #invert Matrix C
Pbar = L*C_inv                          #calculate Matrix Pbar

Pbar1 = Pbar.subs(s1, s/h)              #subsitutes s1 for s/h, with s integration variable
Bij = sp.zeros(4,6)                     #defines Bij Matriz with zeros
Bij[0,:6] = sp.diff(Pbar1[0,:],s)                   #defines first row of Matrix Bij IAW extensions vector
Bij[1,:6] = (Pbar1[1,:]*cosphi+Pbar[0,:]*sinphi)/r  #defines second row of Matrix Bij IAW extensions vector
Bij[2,:6] = -1*sp.diff(sp.diff(Pbar1[1,:],s),s)     #defines third row of Matrix Bij IAW extensions vector
Bij[3,:6] = -1*sp.diff(Pbar1[1,:],s)*sinphi/r       #defines fourth row of Matrix Bij IAW extensions vector
print(Bij)                                          #Prints Bij as defines in book

#Próximos passos: transformar Pbar em P e Bij em B através da Matriz transformação
#Definir método de integração
#Quando estiver tudo definido, converter para função com os símbolos como dados de entrada