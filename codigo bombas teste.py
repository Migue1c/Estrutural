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

C = []
L = []
for s1 in range(2):                                          #for i=0 and i=1
    
    Lk = np.array([[1,s1,0,0,0,0,],[0,0,1,s1,s1**2,s1**3]])     #Li matrix
    L.append(Lk)                                            #apends both Li and Lj vectors
    dw_dsk = np.array([0,0,0,1,2*s1,3*s1**2])                 #bottom row for Ci matrix
    C.append(Lk)                                            #appends both Li and Lj to Ci matrix
    C.append(dw_dsk)                                        #appends dw_dsi to Ci matrix
C  = np.vstack(C)                                           #stacks all vectors vertically for C matrix
L = np.vstack(L)                                            #stacks all vectors vertically for L matrix
C_inv = np.linalg.inv(C)                                    #inverts C matrix

H = np.dot(L,C_inv)                                         #product of matrices L and C_inv to form matrix H -> {v,w}=[H][delta_nodal]
print("Matrix C:\n",C)
print("Matrix L:\n",L)
print("Matrix C_inv:\n",C_inv)
print("Matrix H:\n",H)

#Note that top half of printed matrices has s1=0 and bottom half has s1=1. Refer to book for explanation
#These matrices are for calculation reference only and are not necessarily used



Bij = []                    #opens all vectors necessary to build Bij matrix row by row
BiL1 = []
BiL2 = []
BiL3 = []
BiL4 = []
s = sp.Symbol('s')          #creates symbols for representation. These must be adapted into a function for later use
h = sp.Symbol('h')
r = sp.Symbol('r')
sinphi = sp.Symbol('sinphi')
cosphi = sp.Symbol('cosphi')
vk = np.array([1-(s/h),0,0,(s/h),0,0])                             #opens vectors vk and wk from matrix P manually.
wk = np.array([0,1-3*(s/h)**2+2*(s/h)**3,(s/h)*(1-2*(s/h)+(s/h)**2)*h,0,((s/h)**2)*(3-2*(s/h)),((s/h)**2)*(-1+(s/h))*h])
for i in range(6):
    BiL1.append(sp.diff(vk[(i)],s))                     #creates all four rows from epsilon vector and applies P matrix to them
    BiL2.append((wk[(i)]*cosphi+vk[(i)]*sinphi)/r)
    BiL3.append(-1*sp.diff(sp.diff(wk[(i)],s),s))
    BiL4.append(-1*sp.diff(wk[(i)],s)*sinphi/r)
BiL1 = np.array(BiL1)                                   #turns vectors into horizontal arrays
BiL2 = np.array(BiL2)
BiL3 = np.array(BiL3)
BiL4 = np.array(BiL4)
Bij = np.vstack((BiL1,BiL2,BiL3,BiL4))                  #stacks horizontal arrays to form matrix Bij
print("Matriz Bij:\n",Bij)












